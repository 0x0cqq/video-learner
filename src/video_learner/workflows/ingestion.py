"""将在线视频、本地视频和 Bilibili 缓存统一为可供现有流程读取的本地路径。"""

from collections.abc import Callable
from pathlib import Path

from video_learner.common.core import InputError, output_path
from video_learner.media.download import download_source, source_url

DEFAULT_DOWNLOAD_DIR = Path(".cache/downloads")


def resolve_source(
    source: str | Path,
    download_dir: Path | None = None,
    *,
    output: Path | None = None,
    force: bool = False,
    progress: Callable[[dict], None] | None = None,
) -> Path:
    """本地输入保持只读；URL 先预检下载/输出目录关系，再获取单个视频素材。"""
    url = source_url(source)
    if url is None:
        return Path(source).resolve()
    directory = (download_dir or DEFAULT_DOWNLOAD_DIR).resolve()
    if output is not None:
        target = output.resolve()
        if target.is_relative_to(directory) or directory.is_relative_to(target):
            raise InputError("输出目录不能与下载素材目录重叠；请调整 --download-dir 或 --output")
    directory.mkdir(parents=True, exist_ok=True)
    if output is not None:
        output_path(directory, output, force=force)
    return download_source(url, directory, progress)
