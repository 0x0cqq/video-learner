"""只读媒体导入和按规范时间线解码。"""

import io
import json
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path

import av
from PIL import Image

from video_learner.common.core import US, InputError, TaskError
from video_learner.common.schemas import Source, Track


class OffsetReader(io.RawIOBase):
    def __init__(self, path: Path):
        """建立只读媒体视图，仅在九个 ASCII 0 后的 ftyp box 合法时隐藏该前缀。"""
        self.handle = path.open("rb")
        self.length = path.stat().st_size
        self.offset = 0
        head = self.handle.read(25)
        if head[:9] == b"0" * 9:
            size = int.from_bytes(head[9:13], "big")
            if head[13:17] != b"ftyp" or not 16 <= size <= self.length - 9:
                self.handle.close()
                raise InputError(f"{path.name}：已知前缀后的媒体 box 无效")
            self.offset = 9
        self.seek(0)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        return self.handle.read(size)

    def readinto(self, buffer) -> int:
        return self.handle.readinto(buffer)

    def tell(self) -> int:
        """返回去掉已验证前缀后的逻辑位置，与暴露给解码器的 seek 坐标一致。"""
        return self.handle.tell() - self.offset

    def seek(self, offset: int, whence: int = 0) -> int:
        """将逻辑视图的起点、当前位置或末尾偏移换算成底层文件位置；禁止负位置。"""
        target = offset
        if whence == 1:
            target += self.tell()
        elif whence == 2:
            target += self.length - self.offset
        elif whence != 0:
            raise ValueError("invalid whence")
        if target < 0:
            raise ValueError("negative seek")
        return self.handle.seek(target + self.offset) - self.offset

    def close(self) -> None:
        self.handle.close()
        super().close()


@contextmanager
def open_media(path: Path):
    """同时管理偏移读取器和 PyAV 容器的生命周期，返回容器及实际隐藏的前缀长度。"""
    with OffsetReader(path) as reader:
        with av.open(reader, mode="r") as container:
            yield container, reader.offset


def source_file(source_path: Path, track: Track) -> Path:
    """将轨道中的相对文件名解析到素材根目录内，拒绝清单中的越界路径。"""
    root = source_path if source_path.is_dir() else source_path.parent
    from video_learner.common.core import contained

    return contained(root, track.file)


def track_of(source: Source, kind: str) -> Track:
    """取得唯一的指定类型轨道；缺失或多轨均报错，避免静默选错音视频。"""
    matches = [track for track in source.tracks if track.kind == kind]
    if len(matches) != 1:
        raise InputError(f"需要恰好一个 {kind} 轨道，实际为 {len(matches)}")
    return matches[0]


def inspect_source(path: Path, decode: bool = False, full: bool = False) -> Source:
    """只读探测单视频或单课时缓存，以视频轨道起点建立规范时间线。

    默认只探测轨道；decode 抽查解码，full 顺序解码并核对声明时长。
    opened 表示所有候选媒体均打开并完成轨道探测，与元数据、轨道齐全及解码结果分开。
    """
    path = path.resolve()
    if not path.exists():
        raise InputError("素材路径不存在")
    directory = path.is_dir()
    title = path.stem
    complete = None
    bvid = None
    diagnostics: list[str] = []
    if directory:
        metadata = path / "videoInfo.json"
        if metadata.is_file() and metadata.stat().st_size <= 2_000_000:
            try:
                data = json.loads(metadata.read_text(encoding="utf-8-sig"))
                title = str(data.get("title") or data.get("name") or title)
                bvid = data.get("bvid")
                complete = data.get("isCompleted")
                if data.get("status") == "completed":
                    complete = True
                if not isinstance(complete, bool):
                    complete = None
            except (ValueError, AttributeError):
                diagnostics.append("课程元数据无法解析，标题使用目录名")
        files = sorted(p for p in path.iterdir() if p.is_file())
    else:
        files = [path]
    tracks = []
    opened = True
    for file in files:
        # 联合文件头与媒体扩展名筛选，避免把缓存封面和元数据当作课程视频。
        with file.open("rb") as handle:
            signature = handle.read(32)
        likely_media = (
            not directory
            or signature[4:8] == b"ftyp"
            or signature[:9] == b"0" * 9
            or signature[:4] in (b"\x1aE\xdf\xa3", b"RIFF", b"OggS")
            or file.suffix.lower() in (".m4s", ".mp4", ".mkv", ".mov", ".webm", ".aac")
        )
        if not likely_media:
            continue
        try:
            with open_media(file) as (container, offset):
                for stream in container.streams:
                    if stream.type not in ("video", "audio"):
                        continue
                    tb = stream.time_base
                    if tb is None:
                        raise InputError("轨道缺少 time base")
                    start_pts = stream.start_time or 0
                    duration = (
                        int(stream.duration * tb * US)
                        if stream.duration is not None
                        else int(container.duration or 0)
                    )
                    codec = stream.codec_context
                    tracks.append(
                        Track(
                            file=file.name,
                            index=stream.index,
                            kind=stream.type,
                            codec=codec.name,
                            time_base=str(tb),
                            start_pts=start_pts,
                            start_us=int(start_pts * tb * US),
                            duration_us=duration,
                            prefix_bytes=offset,
                            width=codec.width if stream.type == "video" else None,
                            height=codec.height if stream.type == "video" else None,
                            rate=str(stream.average_rate) if stream.type == "video" else None,
                            sample_rate=codec.sample_rate if stream.type == "audio" else None,
                            channels=codec.channels if stream.type == "audio" else None,
                        )
                    )
        except (av.FFmpegError, InputError, OSError) as exc:
            opened = False
            diagnostics.append(f"{file.name}：媒体无法打开或轨道损坏 ({type(exc).__name__})")
    videos = [t for t in tracks if t.kind == "video"]
    audios = [t for t in tracks if t.kind == "audio"]
    if len(videos) != 1:
        raise InputError("素材需要一个视频轨道；" + "；".join(diagnostics))
    if len(audios) != 1:
        diagnostics.append(f"需要一个音频轨道，发现 {len(audios)} 个")
    video = videos[0]
    if video.duration_us <= 0:
        raise InputError("无法确定视频时长，可能截断或缺失索引")
    root = path if directory else path.parent
    subtitles = sorted(
        p.name
        for p in root.iterdir()
        if p.is_file()
        and p.suffix.lower() in (".srt", ".vtt")
        and (directory or p.stem == path.stem)
    )
    source = Source(
        title=title,
        adapter="bilibili-cache" if directory else "local",
        tracks=tracks,
        duration_us=video.duration_us,
        # 视频起点统一作为零点；音频即使更晚开始，也不能单独归零而丢掉同步偏移。
        origin_us=video.start_us,
        subtitles=subtitles,
        metadata_complete=complete,
        opened=opened,
        diagnostics=diagnostics,
        bvid=bvid,
    )
    if decode or full:
        try:
            for track in tracks:
                with open_media(source_file(path, track)) as (container, _):
                    stream = container.streams[track.index]
                    count = 0
                    if full:
                        last_end = None
                        for frame in container.decode(stream):
                            if frame.is_corrupt or frame.pts is None:
                                raise TaskError("发现损坏帧")
                            at = int(frame.pts * frame.time_base * US)
                            frame_duration = (
                                int(frame.samples * US / frame.sample_rate)
                                if track.kind == "audio"
                                else int(US / stream.average_rate)
                                if stream.average_rate
                                else 0
                            )
                            last_end = at + frame_duration
                            count += 1
                        if (
                            last_end is None
                            or last_end < track.start_us + track.duration_us - US // 4
                        ):
                            # 读到 EOF 不代表完整；容忍少量封装时长差，仍需发现明显提前结束。
                            raise TaskError("解码提前结束，未到达轨道声明时长")
                    else:
                        for fraction in (0, 0.5, 0.95):
                            target = track.start_pts + int(
                                track.duration_us * fraction / US / Fraction(track.time_base)
                            )
                            container.seek(target, stream=stream, backward=True)
                            frame = next(container.decode(stream), None)
                            if frame is None or frame.is_corrupt:
                                raise TaskError("抽查解码无有效帧")
                            count += 1
                    if not count:
                        raise TaskError("未解码出帧")
            source.sampled_decode = True
            source.full_verified = full and not source.diagnostics
        except (av.FFmpegError, TaskError, OSError) as exc:
            source.diagnostics.append(f"解码验证失败 ({type(exc).__name__})")
    return source


def extract_frame(
    path: Path,
    source: Source,
    at_us: int,
    end_us: int | None = None,
    *,
    fallback_start_us: int | None = None,
) -> tuple[Image.Image, int, int]:
    """返回 [at_us, end_us) 内首个有效画面、实际规范微秒时间和原始 PTS。

    先 seek 到目标之前的关键帧再顺序解码，不能将请求时间冒充实际解码时间。
    自动采样可提供 fallback_start_us：没有后续帧时返回该下界内的最后一帧。
    精确换图不启用此选项；解码错误仍直接报告。
    """
    stop = source.duration_us if end_us is None else end_us
    if not 0 <= at_us < stop <= source.duration_us:
        raise InputError("抽帧时间越界")
    if fallback_start_us is not None and not 0 <= fallback_start_us <= at_us:
        raise InputError("采样回取下界越界")
    track = track_of(source, "video")
    with open_media(source_file(path, track)) as (container, _):
        stream = container.streams[track.index]
        target_pts = int(Fraction(at_us + source.origin_us, US) / stream.time_base)
        container.seek(target_pts, stream=stream, backward=True, any_frame=False)
        previous = None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            actual = int(frame.pts * frame.time_base * US) - source.origin_us
            if actual >= stop:
                break
            if actual >= at_us:
                if frame.is_corrupt:
                    raise TaskError("目标画面损坏")
                return frame.to_image(), actual, frame.pts
            if fallback_start_us is not None and actual >= fallback_start_us:
                previous = (frame, actual)
        if previous is not None:
            frame, actual = previous
            if frame.is_corrupt:
                raise TaskError("目标画面损坏")
            return frame.to_image(), actual, frame.pts
    raise TaskError("请求区间没有可用视频帧")
