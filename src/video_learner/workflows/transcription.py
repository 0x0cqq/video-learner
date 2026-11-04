"""共用转写编排：连续切片、识别并发、取消收尾与原始证据保存。"""

import math
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from threading import Event

from video_learner.common.concurrency import ordered_map
from video_learner.common.config import Config
from video_learner.common.core import US, InputError, contained
from video_learner.common.schemas import Source, TranscriptSegment
from video_learner.common.storage import Events, atomic_bytes, write_json
from video_learner.media.audio import encode_wav, pause_cut
from video_learner.providers.asr import QwenASR, Recognizer, validate_qwen_config


def validate_asr_range(config: Config, start_us: int, end_us: int) -> None:
    """在已知素材范围后核对云端切片预算，不重复读取凭据或探测本地后端。"""
    if config.asr_backend == "qwen":
        window = config.asr_window_seconds * US
        if (end_us - start_us + window - 1) // window > config.asr_max_calls:
            raise InputError("Qwen ASR 所需切片数超过 asr_max_calls，请调整范围或配置")


def validate_asr_config(config: Config, start_us: int = 0, end_us: int | None = None) -> None:
    """预检后端和凭据，不下载权重；已知范围时一并核对云端切片预算。"""
    if end_us is not None:
        validate_asr_range(config, start_us, end_us)
    if config.asr_backend == "qwen":
        validate_qwen_config(config)
    else:
        from video_learner.providers.local_asr import validate_local_config

        validate_local_config(config)


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
