"""固定时间范围的音频、转写和候选画面；不包含模型自主工具循环。"""

import re
from pathlib import Path

import av
import numpy as np

from video_learner.common.config import Config
from video_learner.common.core import US, InputError, TaskError, contained, parse_time
from video_learner.common.schemas import FrameEvidence, Source, TranscriptSegment
from video_learner.common.storage import atomic_bytes, digest, write_json
from video_learner.media.io import crop_image, extract_frame, open_media, source_file, track_of


def audio_window(path: Path, source: Source, start_us: int, end_us: int) -> np.ndarray:
    """提取原视频 [start_us, end_us) 的 16 kHz 单声道 float32 波形，最多 310 秒。

    按重采样帧 PTS 放回窗口位置，轨道起点差、时间空洞及片段边缘以静音保留，
    不把解码到的音频简单拼接后误当连续时间。
    """
    if not 0 <= start_us < end_us <= source.duration_us or end_us - start_us > 310 * US:
        raise InputError("音频窗口越界或超过 310 秒内存上限")
    rate = 16000
    result = np.zeros(((end_us - start_us) * rate // US,), dtype=np.float32)
    track = track_of(source, "audio")
    with open_media(source_file(path, track)) as (container, _):
        stream = container.streams[track.index]
        # 提前解码一秒建立重采样器状态，再按 PTS 裁取窗口，减少片段边界偏差。
        target_us = max(track.start_us, start_us + source.origin_us - US)
        target_pts = int(target_us / US / stream.time_base)
        container.seek(target_pts, stream=stream, backward=True)
        resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)
        decoded = False

        def accept(frame):
            """将一个重采样帧按 PTS 放入预分配窗口，仅复制两者相交的采样区间。"""
            if frame.pts is None:
                raise TaskError("重采样输出缺少 PTS，无法保存时间映射")
            absolute_us = int(frame.pts * frame.time_base * US) - source.origin_us
            offset = round((absolute_us - start_us) * rate / US)
            values = frame.to_ndarray().reshape(-1)
            left, right = max(0, offset), min(len(result), offset + len(values))
            if left < right:
                result[left:right] = values[left - offset : right - offset]

        for frame in container.decode(stream):
            if frame.pts is None or frame.is_corrupt:
                raise TaskError("音频帧损坏或缺少 PTS")
            actual = int(frame.pts * frame.time_base * US) - source.origin_us
            if actual > end_us + US:
                break
            decoded = True
            for resampled in resampler.resample(frame):
                accept(resampled)
        for resampled in resampler.resample(None):
            accept(resampled)
        if not decoded:
            raise TaskError("音频窗口未解码出任何帧")
    return result


def load_subtitles(
    path: Path,
    start_us: int,
    end_us: int,
    duration_us: int,
) -> tuple[list[TranscriptSegment], dict]:
    """读取 SRT/VTT 并裁到指定原视频区间，同时保留每条字幕裁剪前的时间。

    覆盖率按时间区间并集计算，重叠字幕不重复计数；时间合法不代表语音同步。
    """
    if not path.is_file():
        raise InputError("字幕文件不存在")
    if path.suffix.lower() not in (".srt", ".vtt") or path.stat().st_size > 20_000_000:
        raise InputError("字幕必须是小于 20 MB 的 SRT/VTT")
    content = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    origin = path.suffix.lower()[1:]
    result = []
    last_start = -1
    for block in re.split(r"\n\s*\n", content):
        lines = block.strip().splitlines()
        timed = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timed is None:
            if lines and not lines[0].startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
                raise InputError("字幕存在无法解析的段落")
            continue
        match = re.fullmatch(r"\s*(\S+)\s+-->\s+(\S+)(?:\s+.*)?", lines[timed])
        if not match:
            raise InputError("字幕时间行无效")
        begin, stop = (parse_time(v.replace(",", ".")) for v in match.groups())
        if not 0 <= begin < stop <= duration_us or begin < last_start:
            raise InputError("字幕时间越界、倒序或为空")
        last_start = begin
        text = "\n".join(lines[timed + 1 :]).strip()
        if not text:
            raise InputError("字幕段落缺少正文")
        if begin < end_us and stop > start_us:
            result.append(
                TranscriptSegment(
                    id=f"tr-{len(result) + 1:06d}",
                    start_us=max(start_us, begin),
                    end_us=min(end_us, stop),
                    text=text,
                    origin=origin,
                    raw_start_us=begin,
                    raw_end_us=stop,
                )
            )
    if not result:
        raise InputError("字幕未覆盖所选片段")
    covered = 0
    previous = start_us
    for segment in result:
        covered += max(0, segment.end_us - max(previous, segment.start_us))
        previous = max(previous, segment.end_us)
    assessment = {
        "coverage_ratio": covered / (end_us - start_us),
        "synchronization": "未人工核对；字幕时间合法不代表与语音同步",
        "first_us": result[0].start_us,
        "last_us": result[-1].end_us,
    }
    return result, assessment


def save_transcript(root: Path, segments: list[TranscriptSegment]) -> None:
    """将转写证据按 UTF-8 JSONL 原子保存，每行保留一个完整的时间映射记录。"""
    atomic_bytes(
        contained(root, "transcript.jsonl"),
        b"".join((segment.model_dump_json() + "\n").encode("utf-8") for segment in segments),
    )


def register_frame(
    path: Path,
    source: Source,
    at_us: int,
    end_us: int,
    root: Path,
    identity: str,
    crop: tuple[int, int, int, int] | None = None,
    parent_id: str | None = None,
) -> FrameEvidence:
    """抽取并保存原帧及裁剪图，登记实际 PTS、规范时间、图片哈希和父证据。

    identity 由应用生成，路径受 root 约束；写入的是工作缓存，成功版本由上层提交。
    """
    image, actual, pts = extract_frame(path, source, at_us, end_us)
    original_path = f".work/frames/{identity}-original.png"
    selected_path = f".work/frames/{identity}.png"
    original = contained(root, original_path)
    original.parent.mkdir(parents=True, exist_ok=True)
    image.save(original, format="PNG")
    crop_image(image, crop).save(contained(root, selected_path), format="PNG")
    return FrameEvidence(
        id=identity,
        at_us=actual,
        requested_us=at_us,
        pts=pts,
        time_base=track_of(source, "video").time_base,
        origin_us=source.origin_us,
        path=selected_path,
        original_path=original_path,
        crop=crop,
        parent_id=parent_id,
        sha256=digest(contained(root, selected_path)),
        original_sha256=digest(original),
    )


def sample_frames(
    path: Path,
    source: Source,
    start_us: int,
    end_us: int,
    config: Config,
    root: Path,
) -> list[FrameEvidence]:
    """固定间隔扫描画面，按变化、章节边界及周期保留候选，并写入采样诊断。

    周期保留用于减少全局灰度差对细小板书变化的漏判；最终插图仍由模型选择。
    """
    frames = []
    previous = None
    previous_chapter = -1
    last_kept_us = None
    scan = []
    for index, at_us in enumerate(range(start_us, end_us, config.sample_seconds * US), 1):
        image, actual, _ = extract_frame(path, source, at_us, end_us)
        selected = crop_image(image, config.crop)
        thumbnail = np.asarray(selected.convert("L").resize((96, 54)), dtype=np.float32) / 255
        change = float(np.mean(np.abs(thumbnail - previous))) if previous is not None else 1.0
        chapter = (at_us - start_us) // (config.chapter_seconds * US)
        # 粉笔笔画可能只改变很少像素；即使整体相似，也周期保留候选供模型选择。
        periodic = last_kept_us is None or at_us - last_kept_us >= 30 * US
        keep = periodic or chapter != previous_chapter or change >= config.image_change_threshold
        scan.append({"requested_us": at_us, "actual_us": actual, "change": change, "kept": keep})
        if keep:
            frames.append(
                register_frame(
                    path,
                    source,
                    at_us,
                    end_us,
                    root,
                    f"frame-{index:06d}",
                    config.crop,
                )
            )
            previous = thumbnail
            previous_chapter = chapter
            last_kept_us = at_us
    write_json(contained(root, ".work/frame-scan.json"), {"samples": scan})
    write_json(contained(root, ".work/frames.json"), {"frames": [f.model_dump() for f in frames]})
    return frames
