"""yt-dlp 单视频适配：下载到调用方目录，返回最小元数据和脱敏进度。"""

import re
from collections.abc import Iterator
from pathlib import Path

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from video_learner.common.core import InputError, TaskError
from video_learner.common.schemas import DownloadMetadata
from video_learner.common.storage import Events


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


def download_video(url: str, directory: Path, events: Events) -> DownloadMetadata:
    """下载单个非直播视频；目录生命周期、媒体检查和发布由应用层负责。

    不读取命令行配置或浏览器登录信息，不保存原始站点信息；失败不透传敏感错误正文。
    """

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
        "outtmpl": str(directory / "%(format_id)s.%(ext)s"),
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
    except DownloadError:
        raise TaskError(
            "yt-dlp 下载失败；请检查网络、站点支持、登录或地区限制，必要时更新 yt-dlp；"
            "也可使用本地视频或 Bilibili 缓存"
        ) from None
    identity_match = re.fullmatch(r"(BV[0-9A-Za-z]{10})(?:_p\d+)?", str(info.get("id", "")))
    return DownloadMetadata(
        title=info.get("title") or "在线视频",
        bvid=identity_match[1] if identity_match else None,
    )
