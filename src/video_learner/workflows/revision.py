"""显式基线、目标范围替换与手改保护。"""

import uuid
from collections.abc import Callable
from pathlib import Path

from video_learner.common.config import load_config
from video_learner.common.core import InputError, contained
from video_learner.common.schemas import NoteBlock, Notebook, ReviewItem
from video_learner.common.storage import (
    Events,
    atomic_bytes,
    directory_lock,
    write_json,
)
from video_learner.exports import DEFAULT_FORMATS, export_book, preflight
from video_learner.media.evidence import register_frame
from video_learner.notes.composition import (
    apply_revision_draft,
    evidence_packet,
    freeze_composition_input,
    validate_notebook,
)
from video_learner.notes.generation import Generator, create_generator
from video_learner.notes.rendering import (
    AnchorConflict,
    frame_of,
    render_block,
)
from video_learner.workflows.versions import (
    baseline_config,
    commit_revision,
    conflict_report,
    extraction_hash,
    load_editable_baseline,
    next_revision,
    verify_source,
)


def revise(
    workdir: Path,
    *,
    base: str = "r001",
    section: str | None = None,
    block: str | None = None,
    instruction: str | None = None,
    at_us: int | None = None,
    config_path: Path | None = None,
    secret: Path | None = None,
    allow_ai_additions: bool | None = None,
    progress: Callable[[dict], None] | None = None,
    generator: Generator | None = None,
    model_provider: str | None = None,
    model: str | None = None,
    pdf_font: Path | None = None,
) -> Path:
    """在显式基线上修订单章或单块，提交包含图片的新独立版本并返回目录。

    先校验快照、当前锚点、提取指纹及源素材；模型只接收目标范围的手改内容。
    按原始字节拼接非目标部分，发布前再次检查并发编辑和源变化，冲突保留诊断。
    精确换图直接使用本地媒体；文字修订才调用图文供应商。
    """
    if (section is None) == (block is None):
        raise InputError("须且只能选择 --section 或 --block")
    image_operation = at_us is not None
    if image_operation and (section or instruction):
        raise InputError("精确换图只接受 --block，不能同时指定文字修订要求")
    if not image_operation and not (instruction and instruction.strip()):
        raise InputError("文字修订必须提供 --instruction")
    preflight(DEFAULT_FORMATS, pdf_font)
    root = workdir.resolve()
    if not root.is_dir():
        raise InputError("工作目录不存在")
    # 先确认工作清单，再使用与 convert 相同的旁路锁，避免在任意输入目录创建工作数据。
    if not contained(root, ".work/manifest.json").is_file():
        raise InputError("此目录没有本工具的工作清单")
    with directory_lock(root.parent / f".{root.name}.lock"):
        editable = load_editable_baseline(root, base)
        manifest, baseline, book = editable.manifest, editable.directory, editable.book
        current, current_hash = editable.current, editable.current_hash
        snapshot, spans, original_spans = editable.snapshot, editable.spans, editable.original_spans
        target_id = section or block
        if target_id not in spans:
            raise InputError("目标 ID 不存在，请查看当前文档中的章节/块 ID")
        span = spans[target_id]
        target_chapter = next(c for c in book.chapters if c.id == (section or span.parent))
        target_block = next((b for b in target_chapter.blocks if b.id == block), None)
        if image_operation and (target_block is None or target_block.kind != "figure"):
            raise InputError("指定块不是图片块")
        original_config = baseline_config(manifest)
        config = load_config(
            config_path,
            base=original_config.model_dump(),
            provider=model_provider,
            model=model,
            instruction=instruction,
            secret_file=str(secret) if secret is not None else None,
            allow_ai_additions=allow_ai_additions,
        )
        if extraction_hash(config) != extraction_hash(original_config):
            # 模型可换，提取设置不能换；当前修订必须继续使用基线对应的同一组证据。
            raise InputError("提取配置与基线不一致，请重新转换到独立目录")
        source_state = verify_source(root, manifest, book)
        source_path, source = source_state.path, source_state.source
        validate_notebook(book, root)
        revision_id, destination = next_revision(root, manifest)
        staging = contained(root, f".work/revision-{uuid.uuid4().hex}.partial")
        staging.mkdir(parents=True, exist_ok=False)
        events = Events(root, progress)
        replacement = b""
        record_id = None
        try:
            with events.stage(f"revise:{revision_id}"):
                if image_operation:
                    old_frame = frame_of(book, target_block.frame_id)
                    if not book.start_us <= at_us < book.end_us:
                        raise InputError("替换帧须在当前转换范围内；其他区间请独立转换")
                    new_frame = register_frame(
                        source_path,
                        source,
                        at_us,
                        book.end_us,
                        root,
                        f"frame-{revision_id}-{target_block.id}",
                    )
                    book.frames.append(new_frame)
                    target_block.frame_id = new_frame.id
                    target_block.evidence_ids = [
                        new_frame.id if identity == old_frame.id else identity
                        for identity in target_block.evidence_ids
                    ]
                    # 保留当前手改图注，只替换受控图片行；无法确定结构时报告冲突。
                    target_current = current[span.start : span.end]
                    target_original = snapshot[
                        original_spans[target_id].start : original_spans[target_id].end
                    ]
                    replacement = replace_figure(
                        target_current, target_original, target_block, book
                    )
                    book.review.append(
                        ReviewItem(
                            reason="已按指定帧更新图片；原图注及相关文字保留，请核对图文一致性",
                            start_us=target_chapter.start_us,
                            end_us=target_chapter.end_us,
                            block_id=target_block.id,
                            evidence_ids=[new_frame.id],
                        )
                    )
                else:
                    active_generator = generator or create_generator(
                        config, events, response_directory=contained(root, ".work/model-responses")
                    )
                    fixed_chapter = target_chapter.model_copy(deep=True)
                    if target_block:
                        from video_learner.notes.rendering import evidence_range

                        ranges = [
                            evidence_range(book, identity) for identity in target_block.evidence_ids
                        ]
                        if ranges:
                            fixed_chapter.start_us = max(book.start_us, min(r[0] for r in ranges))
                            fixed_chapter.end_us = min(book.end_us, max(r[1] for r in ranges) + 1)
                    packet, images = evidence_packet(
                        book,
                        fixed_chapter,
                        config,
                        root,
                        current_markdown=current[span.start : span.end].decode("utf-8"),
                        target_ids=[target_id],
                        current_document=current,
                    )
                    if target_block is not None:
                        packet["target_block_kind"] = target_block.kind
                    if target_block and target_block.kind == "figure":
                        # 图注修订固定原图，不能让模型借此执行 P1 的自然语言重新选图。
                        pinned = frame_of(book, target_block.frame_id)
                        packet["frames"] = [{"id": pinned.id, "at_us": pinned.at_us}]
                        images = [(pinned.id, contained(root, pinned.path))]
                    frozen = freeze_composition_input(packet, images, root)
                    record_id = f"{events.run_id}-{revision_id}"
                    atomic_bytes(contained(root, f".work/revision-inputs/{record_id}.md"), current)
                    write_json(
                        contained(root, f".work/revision-inputs/{record_id}.json"),
                        book.model_dump(),
                    )
                    write_json(
                        contained(root, f".work/composition-inputs/{record_id}.json"),
                        frozen.model_dump(),
                    )
                    draft = active_generator.compose(frozen.packet, images)
                    replacement = apply_revision_draft(
                        book,
                        target_chapter,
                        target_block,
                        target_id,
                        revision_id,
                        frozen.packet,
                        draft,
                    )
                    write_json(
                        contained(root, f".work/composition-drafts/{record_id}.json"),
                        draft.model_dump(),
                    )
                new_current = current[: span.start] + replacement + current[span.end :]
                validate_notebook(book, root)
                export_book(book, root, staging, new_current, base=baseline, pdf_font=pdf_font)
                request_summary = instruction or f"指定帧 {at_us}"
                atomic_bytes(
                    contained(staging, "changes.md"),
                    (
                        f"# {revision_id} 修改摘要\n\n- 基线：{base}\n- 目标：{target_id}\n"
                        f"- 基线当前文稿 SHA-256：{current_hash}\n"
                        f"- 要求：{request_summary}\n"
                        "- 只替换目标范围；非目标 Markdown 字节保持不变。\n"
                        "- 关联证据及待核对事项见 notes.json 与 review.md。\n"
                    ).encode(),
                )
                commit_revision(
                    root,
                    editable,
                    source_state,
                    staging,
                    destination,
                    new_current,
                    {
                        "target": target_id,
                        **({"composition_record": record_id} if record_id else {}),
                    },
                    replacement,
                )
            return destination
        except AnchorConflict as exc:
            report = conflict_report(root, str(exc), replacement)
            raise InputError(f"{exc}；建议片段：{report}") from exc
        except BaseException as exc:
            # 失败暂存目录只供诊断，不能登记为成功修订。
            if staging.exists():
                write_json(
                    contained(staging, "failure.json"),
                    {"status": "failed", "error_type": type(exc).__name__},
                )
            raise


def replace_figure(current: bytes, original: bytes, block: NoteBlock, book: Notebook) -> bytes:
    """仅替换受控图片行，逐字节保留手改图注、附加文字及原有换行格式。

    图片行必须仍与生成快照一致；缺失、重复或手改均报告冲突，不猜测用户意图。
    """
    current_lines = current.splitlines(keepends=True)
    original_lines = original.splitlines(keepends=True)
    rendered_lines = render_block(block, book).splitlines(keepends=True)
    old = [line for line in original_lines if line.startswith(b"![")]
    new = [line for line in rendered_lines if line.startswith(b"![")]
    existing = [i for i, line in enumerate(current_lines) if line.startswith(b"![")]
    if len(old) != 1 or len(new) != 1 or len(existing) != 1:
        raise AnchorConflict("图片块的受控图片行缺失或重复")
    index = existing[0]
    if current_lines[index].rstrip(b"\r\n") != old[0].rstrip(b"\r\n"):
        raise AnchorConflict("图片块的受控图片行已有手改，不能自动替换；请手工合并建议")
    ending = b"\r\n" if current_lines[index].endswith(b"\r\n") else b"\n"
    current_lines[index] = new[0].rstrip(b"\r\n") + ending
    return b"".join(current_lines)
