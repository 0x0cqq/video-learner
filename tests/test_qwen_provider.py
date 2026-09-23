import base64
import json
from types import SimpleNamespace as NS

import pytest
from PIL import Image
from test_provider import VALID, packet

from video_learner.common.config import Config, load_config
from video_learner.common.core import TaskError
from video_learner.common.storage import Events
from video_learner.providers.base import create_provider
from video_learner.providers.qwen import QwenProvider


def chunk(content=None, reasoning=None, finish=None, usage=None):
    """构造正文、思考或仅用量的流式块；usage 尾块刻意不带 choices，以覆盖真实接口形态。"""
    return NS(
        choices=[NS(delta=NS(content=content, reasoning_content=reasoning), finish_reason=finish)]
        if usage is None
        else [],
        usage=usage,
    )


class Stream:
    def __init__(self, chunks):
        self.chunks, self.closed = chunks, False

    def __iter__(self):
        """按顺序交付流式块，遇异常实例就在迭代中抛出，用于模拟收到部分正文后断线。"""
        for value in self.chunks:
            if isinstance(value, Exception):
                raise value
            yield value

    def close(self):
        self.closed = True


class Client:
    def __init__(self, streams):
        self.chat = self.completions = self
        self.streams, self.requests = iter(streams), []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return next(self.streams)


def test_qwen_images_thinking_final_text_and_usage(tmp_path, monkeypatch):
    """验证固定端点、图像编码和思考参数，确认仅持久化最终正文并读取 usage 尾块。"""
    import openai

    picture = tmp_path / "frame.png"
    Image.new("RGB", (8, 8), "red").save(picture)
    stream = Stream(
        [
            chunk(reasoning="private-reasoning-marker"),
            chunk(content=VALID[:40]),
            chunk(content=VALID[40:], finish="stop"),
            chunk(usage=NS(prompt_tokens=120, completion_tokens=90)),
        ]
    )
    client, arguments = Client([stream]), {}

    def constructor(**kwargs):
        arguments.update(kwargs)
        return client

    monkeypatch.setattr(openai, "OpenAI", constructor)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "qwen-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    provider = create_provider(Config(provider="qwen"), Events(tmp_path))
    assert isinstance(provider, QwenProvider)
    assert provider.compose(packet(), [("frame-1", picture)]).title == "章节"
    assert arguments["base_url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert arguments["api_key"] == "qwen-test-key"
    request = client.requests[0]
    assert request["model"] == "qwen3.8-flash"
    assert request["stream"] and request["stream_options"]["include_usage"]
    assert request["extra_body"] == {"enable_thinking": True, "thinking_budget": 1024}
    assert "reasoning_effort" not in request
    assert request["response_format"]["type"] == "json_schema"
    assert "JSON schema:" in request["messages"][0]["content"]
    image_part = request["messages"][1]["content"][-1]
    assert base64.b64decode(image_part["image_url"]["url"].split(",")[1]) == picture.read_bytes()
    saved = list((tmp_path / ".work/model-responses").glob("*.json"))
    assert [path.read_text(encoding="utf-8") for path in saved] == [VALID]
    log = (tmp_path / ".work/logs/events.jsonl").read_text(encoding="utf-8")
    assert "private-reasoning-marker" not in log and "qwen-test-key" not in log
    usage = next(row for row in map(json.loads, log.splitlines()) if row["stage"] == "model_usage")
    assert (usage["input_tokens"], usage["output_tokens"]) == (120, 90)
    assert stream.closed


@pytest.mark.parametrize("finish", [None, "length", "content_filter"])
def test_qwen_incomplete_stream_never_publishes_or_retries(tmp_path, finish):
    """正常 JSON 也不能掩盖流未正常结束；缺停止、截断和过滤三种结果都应拒绝。"""
    stream = Stream([chunk(content=VALID, finish=finish)])
    client = Client([stream])
    with pytest.raises(TaskError, match="未完成"):
        QwenProvider(Config(provider="qwen"), Events(tmp_path), client).compose(packet(), [])
    assert stream.closed and len(client.requests) == 1
    assert not (tmp_path / ".work/model-responses").exists()


def test_qwen_disconnect_discards_partial_answer_and_repairs_schema(tmp_path, monkeypatch):
    """依次模拟半途断线、错误结构和成功响应，防止残片跨请求拼接或修复丢失上下文。"""
    import httpx2
    from openai import APIConnectionError

    monkeypatch.setattr("video_learner.providers.base.time.sleep", lambda _: None)
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
    client = Client(streams)
    provider = QwenProvider(
        Config(provider="qwen", qwen_enable_thinking=False), Events(tmp_path), client
    )
    assert provider.compose(packet(), []).title == "章节"
    assert all(stream.closed for stream in streams)
    assert client.requests[0]["extra_body"] == {"enable_thinking": False}
    assert "结构校验" in client.requests[-1]["messages"][-1]["content"][-1]["text"]
    saved = [
        p.read_text(encoding="utf-8") for p in (tmp_path / ".work/model-responses").glob("*.json")
    ]
    assert sorted(saved) == sorted(["[]", VALID])


def test_provider_switch_resets_stale_model_and_credentials(tmp_path):
    """双向切换供应商时重置继承设置，同时保证本次明确指定的模型和密钥路径优先。"""
    settings = Config(secret_file="deepseek.secret").model_dump()
    qwen = load_config(base=settings, provider="qwen")
    assert (qwen.model, qwen.api_key_env, qwen.secret_file) == (
        "qwen3.8-flash",
        "DASHSCOPE_API_KEY",
        None,
    )
    deepseek = load_config(base=qwen.model_dump(), provider="deepseek")
    assert (deepseek.model, deepseek.api_key_env) == (Config().model, "DEEPSEEK_API_KEY")
    path = tmp_path / "settings.toml"
    path.write_text(
        'provider = "deepseek"\nmodel = "old-model"\nsecret_file = "old.secret"\n', encoding="utf-8"
    )
    selected = load_config(path, provider="qwen", model="explicit-model", secret_file="new.secret")
    assert (selected.model, selected.secret_file) == ("explicit-model", "new.secret")
