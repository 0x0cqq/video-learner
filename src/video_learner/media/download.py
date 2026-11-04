"""用 yt-dlp 下载单个在线视频，发布可供本地媒体流程只读使用的素材目录。"""

import re
import shutil
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from urllib.parse import urlsplit

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from video_learner.common.core import InputError, TaskError, contained
from video_learner.common.schemas import DownloadMetadata
from video_learner.common.storage import Events, write_json
from video_learner.media.io import inspect_source, track_of


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


def select_formats(context: dict) -> Iterator[dict]:
    """优先分别下载最佳画面和音轨；合流素材下载一次，避免为了合并额外依赖 FFmpeg。"""
    formats = list(reversed(context["formats"]))
    video = next(
        (item for item in formats if item.get("vcodec") != "none" and item.get("acodec") == "none"),
        None,
    )
    audio = next(
        (item for item in formats if item.get("vcodec") == "none" and item.get("acodec") != "none"),
        None,
    )
    if video and audio:
        yield {**video, "format_id": "video"}
        yield {**audio, "format_id": "audio"}
    else:
        combined = next(
            (
                item
                for item in formats
                if item.get("vcodec") != "none" and item.get("acodec") != "none"
            ),
            None,
        )
        if combined:
            yield {**combined, "format_id": "media"}


class DownloadLogger:
    """拦住第三方原始日志，避免签名 URL、请求头和站点响应泄漏到终端或事件日志。"""

    def debug(self, message: str) -> None:
        pass

    def warning(self, message: str) -> None:
        pass

    def error(self, message: str) -> None:
        pass


def download_source(
    url: str, directory: Path, progress: Callable[[dict], None] | None = None
) -> Path:
    """下载并探测唯一音视频后原子发布素材；失败或中断清理本次临时目录。

    每次请求独立下载，不复用未完成数据。只持久化标题和 BVID，不保存 yt-dlp 原始信息。
    """
    root = directory.resolve()
    identity = uuid.uuid4().hex
    staging = contained(root, f".{identity}.partial")
    target = contained(root, identity)
    staging.mkdir(parents=True, exist_ok=False)
    events = Events(staging, progress)

    def report(record: dict) -> None:
        """只提取字节计数，忽略 hook 中的 URL、文件名和完整站点信息。"""
        if record["status"] == "downloading":
            events.emit(
                "download",
                "progress",
                completed=record.get("downloaded_bytes", 0),
                total=record.get("total_bytes") or record.get("total_bytes_estimate"),
                unit="bytes",
            )

    options = {
        "format": select_formats,
        "outtmpl": str(staging / "%(format_id)s.%(ext)s"),
        "noplaylist": True,
        "extract_flat": "in_playlist",
        "lazy_playlist": True,
        "logger": DownloadLogger(),
        "quiet": True,
        "noprogress": True,
        "progress_hooks": [report],
        "fixup": "never",
        "cachedir": False,
        "socket_timeout": 30,
        "retries": 2,
        "fragment_retries": 2,
        "abort_on_unavailable_fragments": True,
    }
    try:
        with events.stage("download"):
            with YoutubeDL(options) as downloader:
                info = downloader.extract_info(url, download=False)
                if not info or info.get("_type", "video") != "video":
                    raise InputError("仅支持单视频 URL；请为合集指定具体视频或用 ?p= 选择分 P")
                if info.get("is_live") or info.get("live_status") in (
                    "is_live",
                    "is_upcoming",
                    "post_live",
                ):
                    raise InputError("不支持直播或尚未完成处理的视频，请提供已发布的单视频")
                downloader.process_ie_result(info, download=True)
            identity_match = re.fullmatch(r"(BV[0-9A-Za-z]{10})(?:_p\d+)?", str(info.get("id", "")))
            metadata = DownloadMetadata(
                title=info.get("title") or "在线视频",
                bvid=identity_match[1] if identity_match else None,
            )
            write_json(staging / "download.json", metadata.model_dump())
            source = inspect_source(staging, decode=True)
            track_of(source, "audio")
            if source.diagnostics:
                raise TaskError("下载媒体未通过音视频探测或抽查解码")
            staging.rename(target)
            events.path = target / ".work/logs/events.jsonl"
            events.emit("download_source", "ready", path=str(target))
    except BaseException as exc:
        # 两个路径均由本次 UUID 在指定根目录内创建，不能使用站点返回的路径清理文件。
        for folder in (staging, target):
            if folder.exists():
                shutil.rmtree(contained(root, folder.name))
        if isinstance(exc, DownloadError):
            raise TaskError(
                "yt-dlp 下载失败；请检查网络、站点支持、登录或地区限制，必要时更新 yt-dlp；"
                "也可使用本地视频或 Bilibili 缓存"
            ) from None
        raise
    return target
