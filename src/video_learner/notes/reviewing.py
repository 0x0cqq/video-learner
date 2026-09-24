"""构造独立复审输入，并按明确块目标应用证据约束下的局部修改。"""

from collections.abc import Callable
from difflib import SequenceMatcher
from pathlib import Path

from video_learner.common.config import Config, ReviewStep
from video_learner.common.core import TaskError, contained
from video_learner.common.schemas import (
    Chapter,
    CompositionInput,
    Draft,
    DraftBlock,
    NoteBlock,
    Notebook,
    ReviewItem,
    ReviewResult,
)
from video_learner.notes.composition import (
    chapter_review,
    evidence_packet,
    freeze_composition_input,
    validate_draft,
)
from video_learner.notes.quality import validate_editorial_structure
from video_learner.notes.rendering import expected_spans, render_block

type ReviewFunction = Callable[[dict, list[tuple[str, Path]]], ReviewResult]


def review_input(
    book: Notebook,
    chapter: Chapter,
    config: Config,
    root: Path,
    current: bytes,
    revision_id: str,
    step: ReviewStep | None = None,
) -> tuple[CompositionInput, list[tuple[str, Path]]]:
    """结合实际 Markdown、前后章节与原始证据构造独立复审，不改写手改基线。

    优先保留本章已引用图片，再填入时间均匀的候选，防止复审看不到它要核对的原图。
    """
    spans = expected_spans(current, book)
    span = spans[chapter.id]
    packet, _ = evidence_packet(
        book,
        chapter,
        config,
        root,
        current_markdown=current[span.start : span.end].decode("utf-8"),
    )
    required = {identity for block in chapter.blocks for identity in block.evidence_ids}
    required_frames = [frame for frame in book.frames if frame.id in required]
    if len(required_frames) > config.max_images_per_chapter:
        raise TaskError("本章已引用图片超过基线的送图预算，无法在一次复审中核对全部原图")
    candidates = {item["id"] for item in packet["frames"]} - required
    extra = [frame for frame in book.frames if frame.id in candidates]
    budget = config.max_images_per_chapter - len(required_frames)
    if len(extra) > budget:
        extra = [extra[round(i * (len(extra) - 1) / max(1, budget - 1))] for i in range(budget)]
    frames = required_frames + extra
    frames.sort(key=lambda frame: frame.at_us)
    packet["frames"] = [
        {
            "id": frame.id,
            "at_us": frame.at_us,
            "speech_window_ids": [
                s["id"] for s in packet["transcript"] if s["start_us"] <= frame.at_us < s["end_us"]
            ],
        }
        for frame in frames
    ]
    packet.update(
        operation="review",
        chapter=chapter.model_dump(),
        revision_id=revision_id,
        current_review=[
            item.model_dump()
            for item in book.review
            if item.block_id in {chapter.id, *(block.id for block in chapter.blocks)}
        ],
        next_chapter_not_citable=next(
            (
                {"title": item.title, "beginning": "\n".join(b.body for b in item.blocks)[:3000]}
                for item in book.chapters
                if item.start_us >= chapter.end_us and item.status == "completed"
            ),
            None,
        ),
    )
    if step is not None:
        packet["review_step"] = step.model_dump()
        packet["current_review"] = [
            {"id": index, **item} for index, item in enumerate(packet["current_review"])
        ]
    images = [(frame.id, contained(root, frame.path)) for frame in frames]
    return freeze_composition_input(packet, images, root), images


def validate_review_scope(result: ReviewResult, packet: dict) -> None:
    """限制每轮的写入职能，显式解决疑点；旧记录仍按原单轮契约重放。"""
    if "review_step" not in packet:
        return
    step = ReviewStep.model_validate(packet["review_step"])
    known = {item["id"] for item in packet["current_review"]}
    resolved = result.resolved_review_ids
    if len(set(resolved)) != len(resolved) or not set(resolved) <= known:
        raise TaskError("已解决疑点必须引用本轮 current_review 的有效且不重复的 id")
    edits = [item for item in result.findings if item.action != "report"]
    if step.mode == "report":
        if result.frames is not None or edits or resolved:
            raise TaskError("只报告模式须 frames=null，仅返回 report，不解决或修改已有内容")
        return
    if step.kind == "content" and result.frames is not None:
        raise TaskError("内容复审须 frames=null，保留现有配图")
    if step.kind == "visual" and result.frames is None:
        raise TaskError("图文复审须评估全部候选图")
    blocks = {item["id"]: item for item in packet["chapter"]["blocks"]}
    for edit in edits:
        target = blocks.get(edit.target_id)
        if target is None:
            raise TaskError("复审修改指向不存在的块")
        if step.kind == "content":
            if target["kind"] != "text" or any(b.kind != "text" for b in edit.blocks):
                raise TaskError("内容复审只能修改文字块，图片或图注问题应报告")
        elif (
            target["kind"] != "figure"
            or edit.action != "replace"
            or len(edit.blocks) != 1
            or edit.blocks[0].kind != "figure"
            or edit.blocks[0].frame_id != target["frame_id"]
        ):
            raise TaskError("图文复审的 findings 只能替换同一图片的图注，正文问题应报告")


def reviewed_chapter(result: ReviewResult, packet: dict) -> Chapter:
    """校验复审的范围、引用和逐图决策，确定性地生成修改后的章节。

    修改仅作用于明确的原块，不能利用复审新增无证据事实、重复图片或越界目标。
    """
    validate_review_scope(result, packet)
    chapter = Chapter.model_validate(packet["chapter"])
    edit_id = packet["revision_id"]
    if "review_step" in packet:
        edit_id += "-" + packet["review_step"]["name"]
    blocks = {block.id: block for block in chapter.blocks}
    transcript_ids = {item["id"] for item in packet["transcript"]}
    frame_ids = {item["id"] for item in packet["frames"]}
    available = transcript_ids | frame_ids
    assessed = [item.frame_id for item in result.frames or []]
    if result.frames is not None and (
        len(set(assessed)) != len(assessed) or set(assessed) != frame_ids
    ):
        raise TaskError("复审必须逐张评估本次全部候选图，不能遗漏、重复或新增图片 ID")
    for item in result.frames or []:
        if not set(item.transcript_ids) <= transcript_ids:
            raise TaskError("图片语义对应引用了本次范围之外的语音 ID")
        if item.related_block_id is not None and item.related_block_id not in blocks:
            raise TaskError("图片语义对应的讲义块不存在")
        if item.decision == "use" and item.related_block_id is None:
            raise TaskError("选用图片须指出它支撑的讲义块")
    edits = {}
    for index, finding in enumerate(result.findings, 1):
        if finding.target_id not in blocks:
            raise TaskError("复审修改或疑点指向不存在的块")
        if not finding.evidence_ids or not set(finding.evidence_ids) <= available:
            raise TaskError("复审问题缺少当前章证据或引用越界")
        if finding.action == "report":
            if finding.blocks:
                raise TaskError("待核对项不能同时改写正文")
            continue
        if finding.target_id in edits:
            raise TaskError("同一块有多次复审修改，请合并为一次明确替换")
        if finding.action == "insert_after" and not finding.blocks:
            raise TaskError("插入修改不能为空")
        edits[finding.target_id] = (index, finding)
    updated = []
    for block in chapter.blocks:
        if block.id not in edits:
            updated.append(block)
            continue
        index, finding = edits[block.id]
        if finding.action == "insert_after":
            updated.append(block)
        for position, replacement in enumerate(finding.blocks, 1):
            prefix = "fig" if replacement.kind == "figure" else "blk"
            identity = (
                block.id
                if finding.action == "replace" and position == 1 and replacement.kind == block.kind
                else f"{prefix}-{chapter.id[3:]}-{edit_id}-{index:03d}-{position:02d}"
            )
            updated.append(
                NoteBlock(**replacement.model_dump(), id=identity, chapter_id=chapter.id)
            )
    if not updated:
        raise TaskError("复审不能删除整章全部内容")
    chapter.blocks = (
        position_figures(updated, result, chapter.id, edit_id)
        if result.frames is not None
        else updated
    )
    step = packet.get("review_step")
    validate_draft(
        Draft(
            title=chapter.title,
            blocks=[
                DraftBlock(
                    **{
                        **block.model_dump(include=set(DraftBlock.model_fields)),
                        # 手改块沿用当前 Markdown；旧索引的文体不能冒充用户当前正文。
                        **(
                            {"body": "保留用户手改内容。"}
                            if block.sync_status == "manual_unverified"
                            else {}
                        ),
                    }
                )
                for block in chapter.blocks
            ],
            review=[],
        ),
        packet,
        check_editorial=step is None or (step["kind"] == "content" and step["mode"] == "apply"),
    )
    if step and step["kind"] == "visual" and edits:
        # 图文轮只校验本轮改写的图注，原正文的文体缺陷交给内容轮。
        captions = [block for _, finding in edits.values() for block in finding.blocks]
        validate_editorial_structure(Draft(title="图注", blocks=captions, review=[]), available)
    return chapter


def position_figures(
    updated: list[NoteBlock], result: ReviewResult, chapter_id: str, edit_id: str
) -> list[NoteBlock]:
    """按完整逐图决定增删和排列图片；复用已有图块及图注。"""
    selected = [item for item in result.frames or [] if item.decision == "use"]
    available_figures = {block.frame_id: block for block in updated if block.kind == "figure"}
    # 文字块表示放在解释后；原图片块表示沿用该位置，不猜测它与相邻段落的关系。
    remaining = {block.id for block in updated}
    invalid_links = {
        item.frame_id: item.related_block_id
        for item in selected
        if item.related_block_id not in remaining
    }
    if invalid_links:
        raise TaskError(
            f"选用图片须关联修改后仍保留的块；无效关联：{invalid_links}；"
            f"可关联的保留块：{sorted(remaining)}。请重新关联或将该图片设为 omit"
        )
    figures_after: dict[str, list[NoteBlock]] = {}
    for index, assessment in enumerate(selected, 1):
        figure = available_figures.get(assessment.frame_id)
        if figure is None:
            figure = NoteBlock(
                id=f"fig-{chapter_id[3:]}-{edit_id}-image-{index:02d}",
                chapter_id=chapter_id,
                kind="figure",
                body="",
                category="original",
                evidence_ids=[assessment.frame_id, *assessment.transcript_ids],
                frame_id=assessment.frame_id,
            )
        figures_after.setdefault(assessment.related_block_id, []).append(figure)
    ordered = []
    for block in updated:
        if block.kind == "text":
            ordered.append(block)
        ordered.extend(figures_after.get(block.id, []))
    if not ordered:
        raise TaskError("复审不能删除整章全部内容")
    return ordered


def apply_review_pass(
    book: Notebook, packet: dict, result: ReviewResult, current: bytes
) -> tuple[Notebook, bytes]:
    """纯函数：返回修改后的讲义与 Markdown，保留非目标字节和未解决疑点。

    输入对象保持不变；packet 来自本轮冻结基线，调用方负责顺序执行和保存。
    """
    chapter = reviewed_chapter(result, packet)
    book = book.model_copy(deep=True)
    original = next(item for item in book.chapters if item.id == chapter.id)
    spans = expected_spans(current, book)
    changes = []
    old, new = original.blocks, chapter.blocks
    matcher = SequenceMatcher(a=[b.id for b in old], b=[b.id for b in new], autojunk=False)
    targets = {f.target_id for f in result.findings if f.action == "replace"}
    for tag, a, b, c, d in matcher.get_opcodes():
        if tag == "equal":
            for previous, updated in zip(old[a:b], new[c:d], strict=True):
                if previous.id in targets or render_block(previous, book) != render_block(
                    updated, book
                ):
                    span = spans[previous.id]
                    changes.append(
                        (span.start, span.end, render_block(updated, book).rstrip(b"\n") + b"\n")
                    )
            continue
        start = spans[old[a].id].start if a < len(old) else spans[old[-1].id].end
        end = spans[old[b - 1].id].end if b > a else start
        parts = []
        for block in new[c:d]:
            previous = next((item for item in old if item.id == block.id), None)
            if previous == block and block.id not in targets:
                span = spans[block.id]
                # 移动已有图片时也保留用户图注和换行字节。
                parts.append(current[span.start : span.end])
            else:
                parts.append(render_block(block, book))
        replacement = b"\n".join(part.rstrip(b"\n") for part in parts) + b"\n" if parts else b""
        changes.append((start, end, replacement))
    for start, end, replacement in sorted(changes, reverse=True):
        current = current[:start] + replacement + current[end:]
    targets = {original.id, *(block.id for block in original.blocks)}
    book.chapters[book.chapters.index(original)] = chapter
    remaining_ids = {block.id for block in chapter.blocks}
    if "review_step" not in packet:
        book.review = [item for item in book.review if item.block_id not in targets]
    else:
        resolved = [
            ReviewItem.model_validate({k: v for k, v in item.items() if k != "id"})
            for item in packet["current_review"]
            if item["id"] in result.resolved_review_ids
        ]
        book.review = [item for item in book.review if item not in resolved]
        for item in book.review:
            if item.block_id in targets and item.block_id not in remaining_ids:
                item.block_id = chapter.id
    additions = chapter_review(chapter, [])
    for finding in result.findings:
        if finding.action == "report":
            additions.append(
                ReviewItem(
                    reason=finding.reason,
                    start_us=chapter.start_us,
                    end_us=chapter.end_us,
                    block_id=finding.target_id
                    if finding.target_id in remaining_ids
                    else chapter.id,
                    evidence_ids=finding.evidence_ids,
                )
            )
    for item in additions:
        if item not in book.review:
            book.review.append(item)
    expected_spans(current, book)
    return book, current
