"""共用切片转写流程与阿里云 Qwen 适配；来源按实际音频窗口引用。"""

import base64
import io
import math
import time
import wave
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from threading import Event, Lock
from typing import Protocol

import numpy as np

from video_learner.common.concurrency import ordered_map
from video_learner.common.config import Config
from video_learner.common.core import US, InputError, TaskError, contained
from video_learner.common.schemas import Source, TranscriptSegment
from video_learner.common.storage import Events, atomic_bytes, write_json
from video_learner.providers.base import credential


class Recognizer(Protocol):
    def recognize(self, wav: bytes, cancelled: Event | None = None) -> str: ...

    def close(self) -> None: ...


def validate_asr_config(config: Config, start_us: int, end_us: int) -> None:
    """按显式后端预检；本地识别无需云端凭据，也不会在预检时下载权重。"""
    if config.asr_backend == "qwen":
        validate_qwen_config(config, start_us, end_us)
    else:
        from video_learner.providers.local_asr import validate_local_config

        validate_local_config(config)


def qwen_credential(config: Config) -> str:
    """复用密钥读取规则但替换为 ASR 专用配置，避免误用图文供应商凭据。"""
    return credential(
        config.model_copy(
            update={"api_key_env": config.asr_api_key_env, "secret_file": config.asr_secret_file}
        )
    )


def validate_qwen_config(config: Config, start_us: int, end_us: int) -> None:
    """预检同步识别模型、最低切片调用数和凭据；实际重试仍受总调用上限约束。"""
    if (
        not config.asr_qwen_model.startswith("qwen3-asr-flash")
        or "filetrans" in config.asr_qwen_model
    ):
        raise InputError("Qwen 兼容接口须使用 qwen3-asr-flash 或其快照，不能使用 filetrans")
    window = config.asr_window_seconds * US
    if (end_us - start_us + window - 1) // window > config.asr_max_calls:
        raise InputError("Qwen ASR 所需切片数超过 asr_max_calls，请调整范围或配置")
    if not qwen_credential(config):
        raise InputError("Qwen ASR 缺少 DASHSCOPE_API_KEY 或 --asr-secret 凭据")


def encode_wav(samples: np.ndarray) -> bytes:
    """把最多 60 秒的 16 kHz 单声道浮点波形编码成小端 PCM16 WAV。

    采样率由调用方保证；拒绝空数组和非有限值，幅度裁到 [-1, 1] 后量化。
    """
    if samples.ndim != 1 or not 0 < len(samples) <= 60 * 16000 or not np.isfinite(samples).all():
        raise InputError("Qwen 音频须为最多 60 秒的有限值单声道 16 kHz 波形")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def pause_cut(samples: np.ndarray) -> int:
    """在窗口最后 20% 找至少 300 毫秒的低音量区间，取最靠后的停顿中点。

    使用 20 毫秒 RMS 和固定低音量阈值；无停顿或全段安静时保留完整窗口。
    返回 16 kHz 采样位置，不删除静音，也不推断语义或词句时间。
    """
    step = 320
    count = len(samples) // step
    if count < 15:
        return len(samples)
    rms = np.sqrt(np.mean(samples[: count * step].reshape(count, step) ** 2, axis=1))
    quiet = rms < 0.01
    if quiet.all():
        return len(samples)
    run_start = None
    cut = len(samples)
    for index in range(int(count * 0.8), count + 1):
        if index < count and quiet[index]:
            if run_start is None:
                run_start = index
        elif run_start is not None:
            if index - run_start >= 15:
                cut = (run_start + index) // 2 * step
            run_start = None
    return cut


class QwenASR:
    def __init__(self, config: Config, events: Events, client=None):
        """建立固定 DashScope 端点的独立 ASR 客户端；由本层统计实际请求及重试。"""
        self.config, self.events, self.calls = config, events, 0
        self._call_lock = Lock()
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                api_key=qwen_credential(config),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                max_retries=0,
                timeout=config.request_timeout_seconds,
            )
        self.client = client

    def close(self) -> None:
        self.client.close()

    def recognize(self, wav: bytes, cancelled: Event | None = None) -> str:
        """发送一段 WAV 并返回完整转写文本；临时服务错误有界重试，不完整输出直接失败。

        接口不提供句级时间，因此这里只返回文本，时间映射由音频窗口编排层保存。
        调用编号和总预算在锁内分配；取消会唤醒退避，但不强制中断正在进行的 HTTP 请求。
        """
        from openai import APIConnectionError, APIStatusError, APITimeoutError

        cancelled = cancelled or Event()
        data = "data:audio/wav;base64," + base64.b64encode(wav).decode("ascii")
        options = {"enable_itn": False}
        if self.config.asr_language:
            options["language"] = self.config.asr_language
        for attempt in range(self.config.max_retries + 1):
            with self._call_lock:
                if cancelled.is_set():
                    raise TaskError("Qwen ASR 已取消")
                if self.calls >= self.config.asr_max_calls:
                    raise TaskError("Qwen ASR 已达到调用上限；未完成转写不能发布成功版本")
                self.calls += 1
                call = self.calls
                started = time.monotonic()
                self.events.emit("asr_model_call", "running", call=call, attempt=attempt + 1)
            try:
                response = self.client.chat.completions.create(
                    model=self.config.asr_qwen_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [{"type": "input_audio", "input_audio": {"data": data}}],
                        }
                    ],
                    stream=False,
                    extra_body={"asr_options": options},
                )
                usage = response.usage
                self.events.emit(
                    "asr_model_usage",
                    "received",
                    call=call,
                    model=self.config.asr_qwen_model,
                    seconds=time.monotonic() - started,
                    input_tokens=getattr(usage, "prompt_tokens", None),
                    output_tokens=getattr(usage, "completion_tokens", None),
                    audio_seconds=getattr(usage, "seconds", None),
                )
                if not response.choices or response.choices[0].finish_reason != "stop":
                    raise TaskError("Qwen ASR 输出未完整结束")
                text = response.choices[0].message.content
                if not isinstance(text, str) or len(text) > 20000:
                    raise TaskError("Qwen ASR 返回了无效转写文本")
                self.events.emit("asr_model_call", "completed", call=call)
                return text.strip()
            except APIStatusError as exc:
                self.events.emit(
                    "asr_model_call",
                    "failed",
                    call=call,
                    status_code=exc.status_code,
                    seconds=time.monotonic() - started,
                )
                if exc.status_code not in (408, 409, 429) and exc.status_code < 500:
                    raise InputError(f"Qwen ASR 拒绝请求 (HTTP {exc.status_code})") from None
            except (APIConnectionError, APITimeoutError) as exc:
                self.events.emit(
                    "asr_model_call",
                    "failed",
                    call=call,
                    error_type=type(exc).__name__,
                    seconds=time.monotonic() - started,
                )
            if attempt == self.config.max_retries:
                raise TaskError("Qwen ASR 请求失败，已达到重试上限；超时请求仍可能计费")
            self.events.emit(
                "asr_model_call",
                "retrying",
                attempt=attempt + 1,
                max_retries=self.config.max_retries,
            )
            if cancelled.wait(min(2**attempt, 4)):
                raise TaskError("Qwen ASR 已取消")


def transcribe(
    path: Path,
    source: Source,
    start_us: int,
    end_us: int,
    config: Config,
    root: Path,
    events: Events,
    recognizer: Recognizer | None = None,
) -> list[TranscriptSegment]:
    """按原视频连续且不重叠的窗口识别，保存音频、时间映射和原始响应。

    jobs 仅并发识别请求，切片生成和响应保存保持顺序。每段非空文本只关联实际窗口区间，
    不按字数虚构句级时间；退出前收尾所有工作线程，仅关闭本函数创建的客户端。
    """
    from video_learner.media.evidence import audio_window
    from video_learner.media.io import track_of

    if recognizer is None:
        validate_asr_config(config, start_us, end_us)
    owned_client = recognizer is None
    if recognizer is None:
        if config.asr_backend == "local":
            from video_learner.providers.local_asr import LocalASR

            recognizer = LocalASR(config, events)
        else:
            recognizer = QwenASR(config, events)
    result = []
    size = config.asr_window_seconds * US
    cancelled = Event()

    def windows() -> Iterator[tuple[int, int, str, bytes]]:
        """顺序生成真实切片，最多被并发队列预取 jobs 片，不积存整课波形。"""
        begin, index = start_us, 0
        while begin < end_us and not cancelled.is_set():
            index += 1
            end = min(end_us, begin + size)
            identity = f"audio-{index:05d}"
            with events.stage(f"audio_decode:{identity}"):
                samples = audio_window(path, source, begin, end)
                # 最后一片直接覆盖结尾；其余窗口只向前缩短，不超过服务时长上限。
                cut = pause_cut(samples) if end < end_us else len(samples)
                shortened = cut < len(samples)
                if shortened:
                    end = begin + cut * US // 16000
                samples = samples[:cut]
                events.emit(
                    "audio_boundary",
                    "selected",
                    start_us=begin,
                    end_us=end,
                    reason="pause" if shortened else "limit",
                )
                wav = encode_wav(samples)
                filename = contained(root, f".work/audio/{identity}.wav")
                atomic_bytes(filename, wav)
                write_json(
                    filename.with_suffix(".json"),
                    {
                        "id": identity,
                        "start_us": begin,
                        "end_us": end,
                        "sample_rate": 16000,
                        "source_track": track_of(source, "audio").model_dump(),
                        "origin_us": source.origin_us,
                        "alignment": "audio_window",
                    },
                )
            yield begin, end, identity, wav
            begin = end

    def recognize_window(window: tuple[int, int, str, bytes]) -> tuple[int, int, str, str]:
        """工作线程只识别当前音频，原始响应和转写证据由调用线程按序保存。"""
        begin, end, identity, wav = window
        with events.stage(f"asr_inference:{identity}"):
            text = recognizer.recognize(wav, cancelled=cancelled)
        return begin, end, identity, text

    try:
        events.emit(
            "transcribe",
            "progress",
            completed=0,
            total=end_us - start_us,
            unit="audio",
            request_total=math.ceil((end_us - start_us) / size),
        )
        with closing(ordered_map(recognize_window, windows(), config.jobs, cancelled)) as results:
            for index, (begin, end, identity, text) in enumerate(results, 1):
                write_json(
                    contained(root, f".work/asr-responses/{identity}.json"),
                    {"text": text, "start_us": begin, "end_us": end, "alignment": "audio_window"},
                )
                if text:
                    # 一段文本对应整个真实窗口；没有词句时间戳时不能拆成猜测的细粒度来源。
                    result.append(
                        TranscriptSegment(
                            id=f"tr-{len(result) + 1:06d}",
                            start_us=begin,
                            end_us=end,
                            text=text,
                            origin="asr",
                            alignment="audio_window",
                            chunk_id=identity,
                            raw_start_us=begin,
                            raw_end_us=end,
                        )
                    )
                events.emit(
                    "transcribe",
                    "progress",
                    completed=end - start_us,
                    total=end_us - start_us,
                    unit="audio",
                    request_total=math.ceil((end_us - start_us) * index / (end - start_us)),
                )
    finally:
        if owned_client:
            recognizer.close()
    return result
