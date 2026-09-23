"""本地识别只用替身验证接口和真实媒体编排，不下载权重。"""

from threading import Event
from types import SimpleNamespace

import pytest
from test_conversion import DeterministicProvider

from video_learner.common.config import Config
from video_learner.common.core import TaskError
from video_learner.common.storage import Events
from video_learner.providers.local_asr import LocalASR
from video_learner.workflows.conversion import convert, extraction_hash


def test_local_consumes_lazy_output_and_propagates_failure(tmp_path):
    """真实库在迭代时才推理，故障必须在适配器内暴露；取消不能继续调用模型。"""
    calls = []

    def transcribe(*args, **kwargs):
        calls.append(kwargs)

        def segments():
            yield SimpleNamespace(text="部分")
            raise RuntimeError("GPU failure")

        return segments(), None

    service = LocalASR(
        Config(asr_backend="local"), Events(tmp_path), model=SimpleNamespace(transcribe=transcribe)
    )
    with pytest.raises(TaskError, match="推理失败"):
        service.recognize(b"wav")
    assert calls[0]["vad_filter"] is False
    stop = Event()
    stop.set()
    with pytest.raises(TaskError, match="取消"):
        service.recognize(b"wav", stop)
    assert len(calls) == 1


def test_local_conversion_reuses_windows_without_cloud_key(video, tmp_path, monkeypatch):
    """本地路径复用非零起点的证据与导出，失败/完成均能关闭自己创建的模型。"""
    from video_learner.common.storage import read_json

    closed = []

    class Local:
        def __init__(self, config, events):
            pass

        def recognize(self, wav, cancelled=None):
            return "窗口转写"

        def close(self):
            closed.append(True)

    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setattr("video_learner.providers.local_asr.LocalASR", Local)
    monkeypatch.setattr("video_learner.providers.local_asr.validate_local_config", lambda _: None)
    config = Config(asr_backend="local", jobs=2)
    output = tmp_path / "local"
    convert(video, output, config, start_us=1_000_000, provider=DeterministicProvider())
    book = read_json(output / "notes.json")
    assert book["transcript"][0]["start_us"] == 1_000_000
    assert book["transcript"][0]["alignment"] == "audio_window"
    assert read_json(output / "usage.json")["models"] == []
    assert closed == [True]
    assert extraction_hash(config) != extraction_hash(Config())
    assert extraction_hash(config) == extraction_hash(config.model_copy(update={"jobs": 1}))


def test_local_cancellation_during_iteration_keeps_its_diagnostic(tmp_path):
    """模型迭代期间收到取消，保留取消原因，避免被 RuntimeError 捕获误报为运行库故障。"""
    stop = Event()

    def segments():
        """第一段正常产出后模拟队列取消，再次交付时应立即停止消费。"""
        yield SimpleNamespace(text="第一段")
        stop.set()
        yield SimpleNamespace(text="取消后的内容")

    model = SimpleNamespace(transcribe=lambda *args, **kwargs: (segments(), None))
    service = LocalASR(Config(asr_backend="local"), Events(tmp_path), model=model)
    with pytest.raises(TaskError, match="^本地 ASR 已取消$"):
        service.recognize(b"wav", stop)
