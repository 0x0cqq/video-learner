"""构造独立复审输入，并按明确块目标应用证据约束下的局部修改。"""

from difflib import SequenceMatcher
from pathlib import Path

from video_learner.common.config import Config
from video_learner.common.core import TaskError, contained
from video_learner.common.schemas import (
    Chapter,
    CompositionInput,
    Draft,
    DraftBlock,
    NoteBlock,
    Notebook,
    ReviewItem,
    ReviewPass,
)
from video_learner.notes.composition import (
    chapter_review,
    evidence_packet,
    freeze_composition_input,
    validate_draft,
)
from video_learner.notes.rendering import expected_spans, render_block


def review_input(
    book: Notebook,
    chapter: Chapter,
    config: Config,
    root: Path,
    current: bytes,
    revision_id: str,
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
    images = [(frame.id, contained(root, frame.path)) for frame in frames]
    return freeze_composition_input(packet, images, root), images


def reviewed_chapter(result: ReviewPass, packet: dict) -> Chapter:
    """校验复审的范围、引用和逐图决策，确定性地生成修改后的章节。

    修改仅作用于明确的原块，不能利用复审新增无证据事实、重复图片或越界目标。
    """
    chapter = Chapter.model_validate(packet["chapter"])
    blocks = {block.id: block for block in chapter.blocks}
    transcript_ids = {item["id"] for item in packet["transcript"]}
    frame_ids = {item["id"] for item in packet["frames"]}
    available = transcript_ids | frame_ids
    assessed = [item.frame_id for item in result.frames]
    if len(set(assessed)) != len(assessed) or set(assessed) != frame_ids:
        raise TaskError("复审必须逐张评估本次全部候选图，不能遗漏、重复或新增图片 ID")
    for item in result.frames:
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
                else f"{prefix}-{chapter.id[3:]}-{packet['revision_id']}-{index:03d}-{position:02d}"
            )
            updated.append(
                NoteBlock(**replacement.model_dump(), id=identity, chapter_id=chapter.id)
            )
    if not updated:
        raise TaskError("复审不能删除整章全部内容")
    # 逐图决定是配图的唯一来源，位置和新图块由本地代码生成。
    selected = [item for item in result.frames if item.decision == "use"]
    available_figures = {block.frame_id: block for block in updated if block.kind == "figure"}
    remaining = {block.id for block in updated if block.kind == "text"}
    if not remaining:
        # 没有可用文字的纯图章节，以原图片块作为放置锚点，仍允许复审其视觉内容。
        remaining = {block.id for block in updated}
    if any(item.related_block_id not in remaining for item in selected):
        raise TaskError("选用图片须关联修改后仍保留的文字块")
    figures_after: dict[str, list[NoteBlock]] = {}
    for index, assessment in enumerate(selected, 1):
        figure = available_figures.get(assessment.frame_id)
        if figure is None:
            figure = NoteBlock(
                id=f"fig-{chapter.id[3:]}-{packet['revision_id']}-image-{index:02d}",
                chapter_id=chapter.id,
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
    chapter.blocks = ordered
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
                for block in ordered
            ],
            review=[],
        ),
        packet,
    )
    return chapter


def apply_review_pass(book: Notebook, packet: dict, result: ReviewPass, current: bytes) -> bytes:
    """按块替换 Markdown，保留所有非目标字节，并重建本章的内容疑点。

    packet 来自复审前冻结基线；章节之间的复审输入不受已采用修改影响。
    """
    chapter = reviewed_chapter(result, packet)
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
    book.review = [item for item in book.review if item.block_id not in targets]
    book.review.extend(chapter_review(chapter, []))
    remaining_ids = {block.id for block in chapter.blocks}
    for finding in result.findings:
        if finding.action == "report":
            book.review.append(
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
    expected_spans(current, book)
    return current
