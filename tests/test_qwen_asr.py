import base64
import io
import json
import wave
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock
from types import SimpleNamespace

import numpy as np
import pytest
from openai import APIConnectionError

from video_learner.common.config import Config
from video_learner.common.core import InputError, TaskError
from video_learner.common.storage import Events
from video_learner.media.audio import encode_wav
from video_learner.media.io import inspect_source
from video_learner.providers.asr import QwenASR
from video_learner.workflows.transcription import transcribe, validate_asr_config
from video_learner.workflows.versions import extraction_hash


def client_with(create):
    """提供兼容客户端的最小接口，保证测试不创建网络连接。"""
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_qwen_sends_audio_as_data_and_rejects_incomplete_response(tmp_path):
    """解码捕获的 Base64 WAV 验证音频格式，并确认截断响应不会重试或作为完整转写返回。"""
    received = []

    def create(**kwargs):
        received.append(kwargs)
        return SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(finish_reason="length", message=SimpleNamespace(content="半句"))
            ],
        )

    service = QwenASR(Config(), Events(tmp_path), client=client_with(create))
    wav = encode_wav(np.zeros(16000, dtype=np.float32))
    with pytest.raises(TaskError, match="未完整"):
        service.recognize(wav)
    assert len(received) == 1
    data = received[0]["messages"][0]["content"][0]["input_audio"]["data"]
    with wave.open(io.BytesIO(base64.b64decode(data.split(",", 1)[1])), "rb") as audio:
        assert (audio.getframerate(), audio.getnchannels(), audio.getnframes()) == (16000, 1, 16000)
    assert received[0]["model"] == "qwen3-asr-flash"
    assert received[0]["stream"] is False


def test_qwen_call_budget_and_secret_validation(tmp_path, monkeypatch):
    """验证缺凭据及切片超额预检；模拟网络失败，确认重试也占用 ASR 总调用预算。"""
    import httpx2

    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with pytest.raises(InputError, match="缺少"):
        validate_asr_config(Config(), 0, 1_000_000)
    with pytest.raises(InputError, match="切片数"):
        validate_asr_config(Config(asr_max_calls=1), 0, 61_000_000)

    def create(**kwargs):
        raise APIConnectionError(request=httpx2.Request("POST", "https://example.invalid"))

    service = QwenASR(Config(asr_max_calls=1), Events(tmp_path), client=client_with(create))
    with pytest.raises(TaskError, match="调用上限"):
        service.recognize(encode_wav(np.zeros(16000, dtype=np.float32)))
    assert service.calls == 1


def test_parallel_asr_preserves_order_and_empty_windows(video, tmp_path, monkeypatch):
    """第二片先返回且为空，文本 ID 仍连续、切片关联不重排，原始空响应保留。"""
    source = inspect_source(video)
    source.duration_us = 15_000_000
    second_done = Event()

    def waveform(path, source, begin, end):
        """通过固定振幅标记原始切片编号，同时避免引入停顿切分。"""
        return np.full((end - begin) * 16000 // 1_000_000, (begin / 5e6 + 1) / 10, np.float32)

    class Recognizer:
        def recognize(self, wav, cancelled=None):
            """用信号约束返回顺序，验证保存不依赖线程完成次序。"""
            with wave.open(io.BytesIO(wav), "rb") as audio:
                value = np.frombuffer(audio.readframes(1), dtype="<i2")[0]
            identity = round(value / 32767 * 10)
            if identity == 1:
                assert second_done.wait(3)
            if identity == 2:
                second_done.set()
                return ""
            return str(identity)

    monkeypatch.setattr("video_learner.media.evidence.audio_window", waveform)
    result = transcribe(
        video,
        source,
        0,
        source.duration_us,
        Config(jobs=2, asr_window_seconds=5),
        tmp_path,
        Events(tmp_path),
        recognizer=Recognizer(),
    )
    assert [(s.id, s.chunk_id, s.text, s.start_us, s.end_us) for s in result] == [
        ("tr-000001", "audio-00001", "1", 0, 5_000_000),
        ("tr-000002", "audio-00003", "3", 10_000_000, 15_000_000),
    ]
    saved = json.loads(
        (tmp_path / ".work/asr-responses/audio-00002.json").read_text(encoding="utf-8")
    )
    assert saved == {
        "text": "",
        "start_us": 5_000_000,
        "end_us": 10_000_000,
        "alignment": "audio_window",
    }
    assert extraction_hash(Config(jobs=1)) == extraction_hash(Config(jobs=2))


@pytest.mark.parametrize("mode", ["budget", "retry", "cancel"])
def test_parallel_request_budget_retry_and_cancellation(tmp_path, mode):
    """首批请求强制重叠，分别核对超额拒绝、独立重试编号和退避取消。"""
    import httpx2

    gate, lock, cancelled = Barrier(2), Lock(), Event()
    attempts = 0

    def observe(record):
        """取消场景在退避开始时发出信号，验证不会等待或继续请求。"""
        if mode == "cancel" and record["status"] == "retrying":
            cancelled.set()

    def create(**kwargs):
        """仅首发模拟网络故障，屏障让另一请求同时在途，重试独立进入。"""
        nonlocal attempts
        with lock:
            attempts += 1
            attempt = attempts
        if attempt <= 2:
            gate.wait(timeout=3)
        if mode != "budget" and attempt == 1:
            raise APIConnectionError(request=httpx2.Request("POST", "https://example.invalid"))
        return SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="文本"))
            ],
        )

    events = Events(tmp_path, observe)
    service = QwenASR(
        Config(asr_max_calls=2 if mode == "budget" else 3), events, client=client_with(create)
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(service.recognize, b"wav", cancelled)
            for _ in range(4 if mode == "budget" else 2)
        ]
        results, errors = [], []
        for future in futures:
            try:
                results.append(future.result())
            except TaskError as error:
                errors.append(str(error))
    assert results == ["文本"] * (1 if mode == "cancel" else 2)
    if mode == "budget":
        assert len(errors) == 2 and all("调用上限" in error for error in errors)
    else:
        assert errors == (["Qwen ASR 已取消"] if mode == "cancel" else [])
    count = 3 if mode == "retry" else 2
    assert service.calls == attempts == count and cancelled.is_set() == (mode == "cancel")
    records = [json.loads(line) for line in events.path.read_text(encoding="utf-8").splitlines()]
    started = [e for e in records if e["status"] == "running"]
    assert [e["call"] for e in started] == list(range(1, count + 1))
    assert [e["attempt"] for e in started] == [1, 1] + ([2] if mode == "retry" else [])
    received = {e["call"] for e in records if e["status"] == "received"}
    failed = {e["call"] for e in records if e["status"] == "failed"}
    assert len(received) == len(results)
    assert received.isdisjoint(failed) and received | failed == set(range(1, count + 1))
