"""从现有讲义导出阅读副本；无需模型、原视频或重新生成版本。"""

from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import ValidationError

from video_learner.common.core import InputError, contained, output_path
from video_learner.common.schemas import Notebook
from video_learner.common.storage import digest, directory_lock, read_json, write_json
from video_learner.exports import ExportFormat, prepare_document, write_document
from video_learner.notes.rendering import render_notes
from video_learner.workflows.revision import load_baseline


def export(
    source: Path,
    output: Path,
    formats: list[ExportFormat],
    *,
    base: str | None = None,
    pdf_font: Path | None = None,
) -> Path:
    """读取明确目录或基线的当前文稿，预检资源后原子发布到全新目录。

    支持可携带版本和人工校订副本；保留输入及手改，不登记新的讲义修订版本。
    """
    root = source.resolve()
    target = output_path(root, output)
    if base is not None:
        _, directory, book, snapshot = load_baseline(root, base)
    else:
        directory = root
        try:
            book = Notebook.model_validate(read_json(contained(root, "notes.json")))
        except (OSError, ValueError, ValidationError) as exc:
            raise InputError("输入目录需要有效的 notes.json 和 notes.md") from exc
        snapshot = render_notes(book)
    markdown_path = contained(directory, "notes.md")
    if not markdown_path.is_file() or markdown_path.stat().st_size > 30_000_000:
        raise InputError("需要不超过 30 MB 的 notes.md")
    tracked = [markdown_path, contained(directory, "notes.json")]
    initial = {path: digest(path) for path in tracked}
    current = markdown_path.read_bytes()
    if current != snapshot:
        book.sync_status = "manual_unverified"
    document = prepare_document(book, root, current, base=directory)
    # 导出使用实际文稿和版本图片，即使原视频或缓存已移走，也能独立工作。
    initial.update({path: digest(path) for path in document.images.values()})
    target.parent.mkdir(parents=True, exist_ok=True)
    with directory_lock(target.parent / f".{target.name}.lock"):
        if target.exists():
            raise InputError("输出目录已存在，请指定新目录")
        with TemporaryDirectory(prefix=f".{target.name}-", dir=target.parent) as temporary:
            staging = Path(temporary)
            warnings = write_document(document, staging, formats, pdf_font=pdf_font)
            if any(digest(path) != before for path, before in initial.items()):
                raise InputError("导出期间输入讲义或图片发生变化，请重新导出")
            write_json(
                staging / "export.json",
                {
                    "formats": list(dict.fromkeys(formats)),
                    "title": book.title,
                    "warnings": warnings,
                },
            )
            staging.rename(target)
    return target
