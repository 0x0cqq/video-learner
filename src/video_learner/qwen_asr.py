"""阿里云 Qwen ASR 兼容接口；只返回切片文本，不捏造句级时间戳。"""

import base64
import io
import time
import wave
from pathlib import Path

import numpy as np

from .config import Config
from .core import US, InputError, TaskError, contained
from .provider import credential
from .schemas import Source, TranscriptSegment
from .storage import Events, atomic_bytes, write_json


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


class QwenASR:
    def __init__(self, config: Config, events: Events, client=None):
        """建立固定 DashScope 端点的独立 ASR 客户端；由本层统计实际请求及重试。"""
        self.config, self.events, self.calls = config, events, 0
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                api_key=qwen_credential(config),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                max_retries=0,
                timeout=config.request_timeout_seconds,
            )
        self.client = client

    def recognize(self, wav: bytes) -> str:
        """发送一段 WAV 并返回完整转写文本；临时服务错误有界重试，不完整输出直接失败。

        接口不提供句级时间，因此这里只返回文本，时间映射由音频窗口编排层保存。
        """
        from openai import APIConnectionError, APIStatusError, APITimeoutError

        data = "data:audio/wav;base64," + base64.b64encode(wav).decode("ascii")
        options = {"enable_itn": False}
        if self.config.asr_language:
            options["language"] = self.config.asr_language
        for attempt in range(self.config.max_retries + 1):
            if self.calls >= self.config.asr_max_calls:
                raise TaskError("Qwen ASR 已达到调用上限；未完成转写不能发布成功版本")
            self.calls += 1
            started = time.monotonic()
            self.events.emit("asr_model_call", "running", call=self.calls)
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
                    call=self.calls,
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
                self.events.emit("asr_model_call", "completed", call=self.calls)
                return text.strip()
            except APIStatusError as exc:
                self.events.emit(
                    "asr_model_call", "failed", call=self.calls, status_code=exc.status_code
                )
                if exc.status_code not in (408, 409, 429) and exc.status_code < 500:
                    raise InputError(f"Qwen ASR 拒绝请求 (HTTP {exc.status_code})") from None
            except (APIConnectionError, APITimeoutError) as exc:
                self.events.emit(
                    "asr_model_call", "failed", call=self.calls, error_type=type(exc).__name__
                )
            if attempt == self.config.max_retries:
                raise TaskError("Qwen ASR 请求失败，已达到重试上限；超时请求仍可能计费")
            time.sleep(min(2**attempt, 4))
        raise TaskError("Qwen ASR 请求失败")


def transcribe_qwen(
    path: Path,
    source: Source,
    start_us: int,
    end_us: int,
    config: Config,
    root: Path,
    events: Events,
    recognizer: QwenASR | None = None,
) -> list[TranscriptSegment]:
    """按原视频连续且不重叠的窗口识别，保存音频、时间映射和原始响应。

    每段非空文本只关联实际窗口区间，不按字数虚构句级时间；仅关闭本函数创建的客户端。
    """
    from .evidence import audio_window
    from .media import track_of

    if recognizer is None:
        validate_qwen_config(config, start_us, end_us)
    owned_client = recognizer is None
    recognizer = recognizer or QwenASR(config, events)
    result = []
    size = config.asr_window_seconds * US
    try:
        for index, begin in enumerate(range(start_us, end_us, size), 1):
            end = min(end_us, begin + size)
            identity = f"audio-{index:05d}"
            with events.stage(f"audio_decode:{identity}"):
                samples = audio_window(path, source, begin, end)
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
            with events.stage(f"asr_inference:{identity}"):
                text = recognizer.recognize(wav)
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
    finally:
        if owned_client:
            recognizer.client.close()
    return result
