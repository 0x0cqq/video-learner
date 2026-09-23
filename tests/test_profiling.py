import importlib.util
import json
from pathlib import Path

import pytest
from test_provider import VALID, packet
from test_qwen_provider import Client, Stream, chunk

from video_learner.common.config import Config
from video_learner.common.core import TaskError
from video_learner.common.storage import Events
from video_learner.providers.qwen import QwenProvider


def test_stage_summary_does_not_double_count_or_mix_revisions():
    """嵌套请求只用于归因；失败耗时优先取请求计时，修订重置的调用 ID 不能混入转换。"""
    path = Path(__file__).parents[1] / "tools/profile_conversion.py"
    spec = importlib.util.spec_from_file_location("profile_conversion", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = [
        {"stage": "transcribe", "status": "completed", "seconds": 10},
        {"stage": "plan_chapters", "status": "completed", "seconds": 1},
        {"stage": "asr_model_usage", "status": "received", "seconds": 8},
        {"stage": "asr_model_usage", "status": "received", "seconds": 8},
        {"stage": "audio_decode:audio-1", "status": "completed", "seconds": 2},
        {"stage": "model_usage", "status": "received", "call": 1, "seconds": 3},
        {"stage": "model_call", "status": "failed", "call": 1, "seconds": 3.2},
        {"stage": "model_call", "status": "failed", "call": 2, "seconds": 2},
        {"stage": "model_usage", "status": "received", "call": 3, "seconds": 5},
        {"stage": "compose:ch-001", "status": "completed", "seconds": 11},
        {"stage": "revise:r002", "status": "running"},
        {"stage": "model_usage", "status": "received", "call": 1, "seconds": 100},
    ]
    result = module.summarize_events(records)
    assert result["stage_seconds_total"] == 22
    assert result["model_requests"]["sum_seconds"] == 10
    assert result["failed_model_seconds_measured"] == 5
    assert result["failed_model_requests_without_duration"] == 0
    assert result["audio_prepare_seconds"] == 2
    report = module.render_report({"concurrent": result, "empty": module.summarize_events([])})
    assert "累计 26.0s" in report and "并发请求可重叠" in report
    assert "%" not in report


def test_stream_timing_separates_first_chunks_and_excludes_thinking_text(tmp_path, monkeypatch):
    """用受控时钟验证连接、思考首块和正文首块时间；计数保留而原始思考文本不落盘。"""
    clock = [0.0]
    monkeypatch.setattr("video_learner.providers.qwen.time.monotonic", lambda: clock[0])

    class TimedStream(Stream):
        def __iter__(self):
            """按预设到达时间产生流块，使测试不依赖实际等待或网络。"""
            for at, value in self.chunks:
                clock[0] = at
                yield value

    class TimedClient(Client):
        def create(self, **kwargs):
            clock[0] = 2.0
            return super().create(**kwargs)

    stream = TimedStream(
        [
            (3.0, chunk(reasoning="private-timing-thought")),
            (8.0, chunk(content=VALID)),
            (12.0, chunk(finish="stop")),
        ]
    )
    first = Events(tmp_path)
    provider = QwenProvider(Config(provider="qwen"), first, TimedClient([stream]))
    assert provider.compose(packet(), []).title == "章节"
    second = Events(tmp_path)
    second.emit("next_run", "running")
    raw = (tmp_path / ".work/logs/events.jsonl").read_text(encoding="utf-8")
    records = [json.loads(line) for line in raw.splitlines()]
    timing = next(record for record in records if record["stage"] == "model_stream_timing")
    assert timing["stream_open_seconds"] == 2
    assert timing["first_chunk_seconds"] == timing["first_reasoning_seconds"] == 3
    assert timing["first_answer_seconds"] == 8
    assert timing["seconds"] == 12
    assert timing["answer_characters"] == len(VALID)
    assert timing["reasoning_characters"] == len("private-timing-thought")
    assert timing["status"] == "completed" and stream.closed
    assert "private-timing-thought" not in raw
    assert records[0]["run_id"] != records[-1]["run_id"]
    assert records[-1]["elapsed_seconds"] == 0
    assert all("timestamp_utc" in record for record in records)
    response = next(record for record in records if record["stage"] == "model_response")
    assert (tmp_path / f".work/model-responses/{response['response_id']}.json").is_file()


def test_stream_service_error_is_measured_without_leaking_or_retrying(tmp_path):
    """服务在流内报错时记录未完成计时；即使先收到 stop，也不把失败标为成功或重试。"""
    import httpx2
    from openai import APIError

    stream = Stream(
        [
            chunk(content=VALID, finish="stop"),
            APIError(
                "private-provider-message",
                request=httpx2.Request("POST", "https://example.invalid"),
                body={},
            ),
        ]
    )
    client = Client([stream])
    provider = QwenProvider(Config(provider="qwen"), Events(tmp_path), client)
    with pytest.raises(TaskError, match="不自动重试") as error:
        provider.compose(packet(), [])
    records = [
        json.loads(line)
        for line in (tmp_path / ".work/logs/events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert "private-provider-message" not in str(error.value) + json.dumps(records)
    timing = next(record for record in records if record["stage"] == "model_stream_timing")
    assert timing["status"] == "incomplete"
    assert records[-1]["stage"] == "model_call" and records[-1]["status"] == "failed"
    assert records[-1]["seconds"] >= 0
    assert stream.closed and len(client.requests) == 1
