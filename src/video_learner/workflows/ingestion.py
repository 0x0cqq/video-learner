"""将在线视频、本地视频和 Bilibili 缓存统一为可供现有流程读取的本地路径。"""

import re
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

from video_learner.common.core import InputError, TaskError, contained, output_path
from video_learner.common.schemas import Source
from video_learner.common.storage import Events, write_json
from video_learner.media.io import inspect_source, track_of

DEFAULT_DOWNLOAD_DIR = Path(".cache/downloads")


def source_url(value: str | Path) -> str | None:
    """识别 HTTP(S) URL 或裸 BVID；本地路径不改写，保留 URL 的分 P 和查询参数。"""
    if isinstance(value, Path):
        return None
    if re.fullmatch(r"BV[0-9A-Za-z]{10}", value):
        return f"https://www.bilibili.com/video/{value}"
    if "://" not in value:
        return None
    try:
        parsed = urlsplit(value)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError
        if parsed.username or parsed.password:
            raise ValueError
        _ = parsed.port
    except ValueError as exc:
        raise InputError("在线视频须为无内嵌账号密码的 HTTP(S) URL，或完整 BVID") from exc
    return value


def prepare_source(
    source: str | Path,
    download_dir: Path | None = None,
    *,
    output: Path | None = None,
    force: bool = False,
    progress: Callable[[dict], None] | None = None,
    decode: bool = False,
    full: bool = False,
) -> tuple[Path, Source]:
    """统一取得素材路径和探测结果；在线素材由本层验证后发布，失败清理本次数据。

    本地输入保持只读，且不加载下载适配器；URL 先预检下载/输出关系，至少抽查解码一次。
    """
    url = source_url(source)
    if url is None:
        path = Path(source).resolve()
        if output is not None:
            output_path(path, output, force=force)
        return path, inspect_source(path, decode=decode, full=full)
    from video_learner.media.download import download_video

    directory = (download_dir or DEFAULT_DOWNLOAD_DIR).resolve()
    if output is not None:
        target = output.resolve()
        if target.is_relative_to(directory) or directory.is_relative_to(target):
            raise InputError("输出目录不能与下载素材目录重叠；请调整 --download-dir 或 --output")
    directory.mkdir(parents=True, exist_ok=True)
    if output is not None:
        output_path(directory, output, force=force)
    identity = uuid.uuid4().hex
    staging = contained(directory, f".{identity}.partial")
    target = contained(directory, identity)
    staging.mkdir(parents=True, exist_ok=False)
    try:
        events = Events(staging, progress)
        with events.stage("download"):
            metadata = download_video(url, staging, events)
            write_json(staging / "download.json", metadata.model_dump())
            inspected = inspect_source(staging, decode=True, full=full)
            track_of(inspected, "audio")
            if inspected.diagnostics:
                raise TaskError("下载媒体未通过音视频探测或解码检查")
            staging.rename(target)
            events.path = target / ".work/logs/events.jsonl"
            events.emit("download_source", "ready", path=str(target))
    except BaseException:
        # 只清理本次 UUID 下的目录；日志初始化失败同样不能留下半成品。
        for folder in (staging, target):
            if folder.exists():
                shutil.rmtree(contained(directory, folder.name))
        raise
    return target, inspected
