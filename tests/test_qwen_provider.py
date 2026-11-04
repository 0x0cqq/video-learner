import pytest
from model_fakes import VALID, Stream, chunk, packet, session

from video_learner.common.core import TaskError
from video_learner.common.usage import summarize_usage


@pytest.mark.parametrize("finish", [None, "length"])
def test_qwen_incomplete_stream_never_publishes_or_retries(tmp_path, finish):
    """正常 JSON 也不能掩盖流未正常结束；缺少停止标记或截断时应拒绝发布。"""
    stream = Stream([chunk(content=VALID, finish=finish)])
    service, client = session(tmp_path, "qwen", [stream])
    with pytest.raises(TaskError, match="未完成"):
        service.compose(packet(), [])
    assert stream.closed and len(client.requests) == 1
    assert not (tmp_path / ".work/model-responses").exists()


def test_qwen_disconnect_discards_partial_answer_and_repairs_schema(tmp_path, monkeypatch):
    """依次模拟半途断线、错误结构和成功响应，防止残片跨请求拼接或修复丢失上下文。"""
    import httpx2
    from openai import APIConnectionError

    monkeypatch.setattr("video_learner.notes.generation.time.sleep", lambda _: None)
    streams = [
        Stream(
            [
                chunk(content="corrupt-prefix"),
                APIConnectionError(request=httpx2.Request("POST", "https://example.invalid")),
            ]
        ),
        Stream([chunk(content="[]", finish="stop")]),
        Stream([chunk(content=VALID, finish="stop")]),
    ]
    provider, client = session(tmp_path, "qwen", streams, qwen_enable_thinking=False)
    assert provider.compose(packet(), []).title == "章节"
    assert all(stream.closed for stream in streams)
    assert client.requests[0]["extra_body"] == {"enable_thinking": False}
    assert "结构校验" in client.requests[-1]["messages"][-1]["content"][-1]["text"]
    saved = [
        p.read_text(encoding="utf-8") for p in (tmp_path / ".work/model-responses").glob("*.json")
    ]
    assert sorted(saved) == sorted(["[]", VALID])


@pytest.mark.parametrize("failure", ["service", "cancelled"])
def test_qwen_failure_preserves_already_received_usage(tmp_path, failure):
    """服务端计量先于流错误或用户中断到达时，失败调用仍保留已知计费信息。"""
    from types import SimpleNamespace as NS

    import httpx2
    from openai import APIError

    error = (
        KeyboardInterrupt()
        if failure == "cancelled"
        else APIError(
            "private-error", request=httpx2.Request("POST", "https://example.invalid"), body=None
        )
    )
    stream = Stream(
        [
            chunk(content=VALID, finish="stop"),
            chunk(
                usage=NS(
                    prompt_tokens=1234,
                    completion_tokens=567,
                    prompt_tokens_details=NS(cached_tokens=100),
                )
            ),
            error,
        ]
    )
    service, client = session(tmp_path, "qwen", [stream], max_retries=0)
    with pytest.raises(KeyboardInterrupt if failure == "cancelled" else TaskError):
        service.compose(packet(), [])
    report = summarize_usage(service.events.usage_events, service.config)
    assert report["input_tokens"] == 1234 and report["output_tokens"] == 567
    assert report["models"][0]["cached_input_tokens"] == 100
    assert report["missing_token_usage"] == 0
    assert stream.closed and len(client.requests) == 1
