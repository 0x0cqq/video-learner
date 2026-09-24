"""三个并列导出适配器；准备文档与选择格式共用，版本发布由工作流负责。"""

from pathlib import Path
from typing import Literal

from video_learner.common.core import InputError, contained
from video_learner.common.schemas import Notebook
from video_learner.common.storage import write_json
from video_learner.exports.document import ExportDocument, prepare_document

type ExportFormat = Literal["markdown", "html", "pdf"]


def write_document(
    document: ExportDocument,
    destination: Path,
    formats: list[ExportFormat],
    *,
    pdf_font: Path | None = None,
) -> list[str]:
    """将同一份只读快照交给选中的适配器，返回需要用户注意的排版降级。"""
    from video_learner.exports import html, markdown, pdf

    adapters = {"markdown": markdown.write, "html": html.write, "pdf": pdf.write}
    if not formats or any(value not in adapters for value in formats):
        raise InputError("导出格式须为 markdown、html 或 pdf")
    warnings = []
    for name in dict.fromkeys(formats):
        options = {"font": pdf_font} if name == "pdf" else {}
        warnings.extend(adapters[name](document, destination, **options))
    write_json(contained(destination, "notes.json"), document.book.model_dump())
    write_json(contained(destination, "source.json"), document.book.source.model_dump())
    return list(dict.fromkeys(warnings))


def export_book(
    book: Notebook,
    evidence_root: Path,
    destination: Path,
    markdown: bytes | None = None,
    *,
    base: Path | None = None,
) -> bytes:
    """转换和修订保存可继续编辑的 Markdown 版本及结构化索引。"""
    document = prepare_document(book, evidence_root, markdown, base=base)
    write_document(document, destination, ["markdown"])
    return document.markdown
