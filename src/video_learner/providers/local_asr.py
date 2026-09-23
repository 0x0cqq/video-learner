"""可选 faster-whisper 适配器；复用 WAV 切片，不启动本地服务或自动回退。"""

import io
import os
import time
from importlib.util import find_spec
from threading import Event

from video_learner.common.config import Config
from video_learner.common.core import InputError, TaskError
from video_learner.common.storage import Events


def local_model(config: Config) -> str:
    """默认 CPU 使用 small，CUDA 使用多语种 large-v3-turbo；允许显式模型目录。"""
    return config.asr_local_model or ("large-v3-turbo" if config.asr_device == "cuda" else "small")


def validate_local_config(config: Config) -> None:
    """只检查可选依赖和资源配置；权重在实际识别阶段按需加载。"""
    if find_spec("faster_whisper") is None:
        raise InputError("本地 ASR 需要可选依赖：uv sync --extra local-asr")
    if config.asr_local_model is not None and not config.asr_local_model.strip():
        raise InputError("asr_local_model 必须为模型名或本地模型目录")
    if config.asr_device == "cuda" and config.jobs != 1:
        raise InputError("本地 CUDA 为控制显存使用单 worker，请设置 jobs=1")
    if config.asr_device == "cuda" and os.name == "nt":
        import ctypes

        try:
            # 提前报告缺失 DLL，避免原生推理库在首次卷积时直接终止进程。
            for name in ("cublas64_12.dll", "cudnn64_9.dll"):
                # 与 CTranslate2 的 LoadLibrary 使用同一 PATH 搜索规则。
                ctypes.WinDLL(name, winmode=0)
        except OSError:
            raise InputError(
                "CUDA 需要 CUDA 12/cuDNN 9 DLL；请按 docs/usage.md 将运行库 bin 加入 PATH"
            ) from None


class LocalASR:
    def __init__(self, config: Config, events: Events, model=None):
        """每次转写加载一次模型；CTranslate2 的 worker 复用权重并释放 Python GIL。"""
        self.config, self.events = config, events
        compute_type = "int8_float16" if config.asr_device == "cuda" else "int8"
        with events.stage("asr_model_load"):
            if model is None:
                validate_local_config(config)
                try:
                    from faster_whisper import WhisperModel

                    model = WhisperModel(
                        local_model(config),
                        device=config.asr_device,
                        compute_type=compute_type,
                        cpu_threads=config.asr_cpu_threads,
                        num_workers=config.jobs,
                    )
                except (ImportError, OSError, RuntimeError, ValueError) as exc:
                    raise InputError(
                        f"本地 ASR 模型加载失败（{type(exc).__name__}）；请检查权重、"
                        "设备及 CUDA 12/cuDNN 9 运行库，详见 docs/usage.md"
                    ) from None
            self.model = model
        events.emit(
            "asr_local_config",
            "ready",
            model=local_model(config),
            device=config.asr_device,
            compute_type=compute_type,
            cpu_threads=config.asr_cpu_threads,
            workers=config.jobs,
            beam_size=config.asr_beam_size,
        )

    def recognize(self, wav: bytes, cancelled: Event | None = None) -> str:
        """消费完整推理迭代器后返回窗口文本；保留静音，沿用实际切片来源精度。"""
        if cancelled and cancelled.is_set():
            raise TaskError("本地 ASR 已取消")
        started = time.monotonic()
        try:
            segments, _ = self.model.transcribe(
                io.BytesIO(wav),
                language=self.config.asr_language,
                beam_size=self.config.asr_beam_size,
                temperature=0,
                condition_on_previous_text=False,
                vad_filter=False,
                word_timestamps=False,
            )
            parts = []
            for segment in segments:
                if cancelled and cancelled.is_set():
                    raise TaskError("本地 ASR 已取消")
                parts.append(segment.text)
            text = "".join(parts).strip()
        except TaskError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            raise TaskError(
                f"本地 ASR 推理失败（{type(exc).__name__}），请检查模型及运行库"
            ) from None
        self.events.emit("asr_local_inference", "completed", seconds=time.monotonic() - started)
        return text

    def close(self) -> None:
        """共用队列收尾后卸载权重，释放 CPU/GPU 内存。"""
        self.model.model.unload_model()
