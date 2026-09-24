"""Markdown 适配器保留锚点、手改原字节和可携带图片。"""

import shutil
from pathlib import Path

from video_learner.common.core import contained
from video_learner.common.storage import atomic_bytes
from video_learner.exports.document import ExportDocument


def write(document: ExportDocument, destination: Path) -> list[str]:
    """写入三份 Markdown 和所有引用图片，保留原始换行与手改格式。"""
    for relative, original in document.images.items():
        target = contained(destination, relative)
        if target != original:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
    for part in document.parts:
        atomic_bytes(contained(destination, part.name + ".md"), part.markdown)
    return []
