"""有界音频窗口的 WAV 编码与停顿切点选择。"""

import io
import wave

import numpy as np

from video_learner.common.core import InputError


def encode_wav(samples: np.ndarray) -> bytes:
    """把最多 60 秒的 16 kHz 单声道浮点波形编码成小端 PCM16 WAV。

    采样率由调用方保证；拒绝空数组和非有限值，幅度裁到 [-1, 1] 后量化。
    """
    if samples.ndim != 1 or not 0 < len(samples) <= 60 * 16000 or not np.isfinite(samples).all():
        raise InputError("音频须为最多 60 秒的有限值单声道 16 kHz 波形")
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
