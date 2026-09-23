"""基于已保存的证据和模型草稿，离线重建转换结果。"""

from pathlib import Path

from video_learner.common.config import Config
from video_learner.common.core import InputError, TaskError, contained
from video_learner.common.schemas import CompositionInput, Draft, Notebook
from video_learner.common.storage import digest, read_json
from video_learner.notes.composition import (
    apply_chapter_draft,
    apply_revision_draft,
    composition_input,
    validate_notebook,
)
from video_learner.notes.rendering import expected_spans, render_notes


def replay_conversion(workdir: Path) -> Notebook:
    """不用视频或模型重建 r001，核对逐章输入、草稿和生成快照。"""
    root = workdir.resolve()
    manifest = read_json(contained(root, ".work/manifest.json"))
    if manifest.get("status") != "completed":
        raise InputError("仅能离线重放已完成的转换")
    evidence_path = contained(root, ".work/evidence.json")
    if not evidence_path.is_file():
        raise InputError("该转换没有冻结的模型前证据，无法离线重建")
    config = Config.model_validate(manifest["config"])
    book = Notebook.model_validate(read_json(evidence_path))
    for chapter in book.chapters:
        recorded = CompositionInput.model_validate(
            read_json(contained(root, f".work/composition-inputs/{chapter.id}.json"))
        )
        rebuilt, _ = composition_input(book, chapter, config, root)
        if rebuilt != recorded:
            raise TaskError(f"{chapter.id} 的模型输入与冻结记录不一致")
        draft = Draft.model_validate(
            read_json(contained(root, f".work/composition-drafts/{chapter.id}.json"))
        )
        apply_chapter_draft(book, chapter, recorded.packet, draft)
    validate_notebook(book, root)
    if book.model_dump() != read_json(contained(root, "notes.json")):
        raise TaskError("离线重建的讲义索引与原始版本不一致")
    generated = contained(root, ".work/versions/r001.generated.md").read_bytes()
    if render_notes(book) != generated:
        raise TaskError("离线重建的 Markdown 与原始生成快照不一致")
    return book


def replay_revision(workdir: Path, revision_id: str) -> Notebook:
    """从文字修订时的手改基线、模型输入和草稿离线重建指定版本。"""
    root = workdir.resolve()
    manifest = read_json(contained(root, ".work/manifest.json"))
    version = manifest.get("versions", {}).get(revision_id)
    record_id = version.get("composition_record") if version else None
    if not record_id:
        raise InputError("指定版本没有可重放的文字修订记录")
    book = Notebook.model_validate(
        read_json(contained(root, f".work/revision-inputs/{record_id}.json"))
    )
    current = contained(root, f".work/revision-inputs/{record_id}.md").read_bytes()
    frozen = CompositionInput.model_validate(
        read_json(contained(root, f".work/composition-inputs/{record_id}.json"))
    )
    for image in frozen.images:
        if digest(contained(root, image.path)) != image.sha256:
            raise TaskError("修订时使用的图片证据已经变化")
    target_id = version["target"]
    spans = expected_spans(current, book)
    if target_id not in spans:
        raise TaskError("修订目标在冻结基线中不存在")
    span = spans[target_id]
    if current[span.start : span.end].decode("utf-8") != frozen.packet["current_markdown"]:
        raise TaskError("修订目标与冻结的模型输入不一致")
    chapter = next(
        (
            item
            for item in book.chapters
            if item.id == target_id or any(b.id == target_id for b in item.blocks)
        ),
        None,
    )
    if chapter is None:
        raise TaskError("修订目标的章节不存在")
    block = next((item for item in chapter.blocks if item.id == target_id), None)
    draft = Draft.model_validate(
        read_json(contained(root, f".work/composition-drafts/{record_id}.json"))
    )
    replacement = apply_revision_draft(
        book, chapter, block, target_id, revision_id, frozen.packet, draft
    )
    generated = current[: span.start] + replacement + current[span.end :]
    validate_notebook(book, root)
    expected_spans(generated, book)
    if generated != contained(root, f".work/versions/{revision_id}.generated.md").read_bytes():
        raise TaskError("离线重建的修订 Markdown 与生成快照不一致")
    directory = contained(root, version["folder"])
    if book.model_dump() != read_json(contained(directory, "notes.json")):
        raise TaskError("离线重建的修订索引与版本不一致")
    return book
