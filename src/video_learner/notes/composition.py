"""预先登记章节覆盖，按固定证据包组织和验证讲义。"""

from fractions import Fraction
from pathlib import Path

from PIL import Image

from video_learner.common.config import Config
from video_learner.common.core import US, TaskError, contained
from video_learner.common.schemas import (
    Chapter,
    Draft,
    FrameEvidence,
    NoteBlock,
    Notebook,
    ReviewItem,
    TranscriptSegment,
)
from video_learner.common.storage import digest
from video_learner.providers.base import Provider


def plan_chapters(
    start_us: int,
    end_us: int,
    config: Config,
    transcript: list[TranscriptSegment] | None = None,
) -> list[Chapter]:
    """按目标章长划分，在附近 20% 范围优先采用转写边界，避免同一切片跨章重复。

    这只是连续证据分组，不声称停顿就是话题边界；过长或重叠字幕仍可按长度分组。
    """
    size = config.chapter_seconds * US
    boundaries = sorted({s.end_us for s in transcript or []})
    chapters = []
    begin = start_us
    while begin < end_us:
        stop = min(end_us, begin + size)
        if stop < end_us:
            nearby = [b for b in boundaries if abs(b - stop) <= size // 5 and b < end_us]
            if nearby:
                stop = min(nearby, key=lambda b: abs(b - stop))
            # 极短尾章容易把半句话扩写成整节；合并仍保留完整连续范围。
            if end_us - stop <= size // 5:
                stop = end_us
        index = len(chapters) + 1
        chapters.append(
            Chapter(
                id=f"ch-{index:03d}",
                title=f"章节 {index}",
                start_us=begin,
                end_us=stop,
            )
        )
        begin = stop
    return chapters


def evidence_packet(
    book: Notebook,
    chapter: Chapter,
    config: Config,
    root: Path,
    current_markdown: str | None = None,
    target_ids: list[str] | None = None,
) -> tuple[dict, list[tuple[str, Path]]]:
    """组装目标范围的可引用证据及受控本地图片路径，供转换或局部修订使用。

    相邻十秒转写与上一章末尾仅帮助衔接，不可引用；图片超限时均匀取样，避免丢掉章尾。
    """
    context_start = max(book.start_us, chapter.start_us - 10 * US)
    context_end = min(book.end_us, chapter.end_us + 10 * US)
    segments = [
        s for s in book.transcript if s.start_us < chapter.end_us and s.end_us > chapter.start_us
    ]
    context = [
        s.text
        for s in book.transcript
        if s.start_us < context_end and s.end_us > context_start and s not in segments
    ]
    frames = [f for f in book.frames if chapter.start_us <= f.at_us < chapter.end_us]
    if len(frames) > config.max_images_per_chapter:
        # 在整章均匀保留有界候选，直接截取前几张会丢掉章尾内容。
        maximum = config.max_images_per_chapter
        indices = [round(i * (len(frames) - 1) / max(1, maximum - 1)) for i in range(maximum)]
        frames = [frames[i] for i in indices]
    packet = {
        "operation": "revise" if current_markdown is not None else "convert",
        "title": book.title,
        "chapter_id": chapter.id,
        "start_us": chapter.start_us,
        "end_us": chapter.end_us,
        "profile": config.profile,
        "instruction": config.instruction,
        "allow_ai_additions": config.allow_ai_additions,
        "transcript": [s.model_dump() for s in segments],
        "adjacent_context_not_citable": context,
        "previous_chapter_not_citable": next(
            (
                {"title": c.title, "ending": "\n".join(b.body for b in c.blocks)[-3000:]}
                for c in reversed(book.chapters)
                if c.end_us <= chapter.start_us and c.status == "completed"
            ),
            None,
        ),
        "frames": [{"id": f.id, "at_us": f.at_us} for f in frames],
        "current_markdown": current_markdown,
        "target_ids": target_ids,
    }
    return packet, [(f.id, contained(root, f.path)) for f in frames]


def validate_draft(draft: Draft, packet: dict) -> None:
    """检查草稿的证据边界、块类型、补充授权及正文结构；违规抛 TaskError。

    检查的是引用和结构合法性，不能据此认定正文已被原课事实支持。
    """
    transcripts = {s["id"]: s for s in packet["transcript"]}
    frames = {f["id"]: f for f in packet["frames"]}
    available = transcripts.keys() | frames.keys()
    selected_frames = set()
    for block in draft.blocks:
        if block.category == "ai_addition" and not packet["allow_ai_additions"]:
            raise TaskError("模型返回了未经用户要求的 AI 补充")
        if block.category != "ai_addition" and not block.evidence_ids:
            raise TaskError("课程内容块缺少来源证据")
        if not set(block.evidence_ids) <= available:
            raise TaskError("模型引用了本次证据包以外的 ID")
        if block.kind == "figure":
            if block.frame_id not in frames or block.frame_id not in block.evidence_ids:
                raise TaskError("图片块未引用已提供的图像证据")
            if block.frame_id in selected_frames:
                raise TaskError("同章重复使用同一个截图，请合并图片块")
            selected_frames.add(block.frame_id)
        else:
            if block.frame_id is not None:
                raise TaskError("文字块不能指定图片")
            if not block.body.strip():
                raise TaskError("文字块正文不能为空；图片无需图注时可用空正文")
        if block.evidence_ids:
            local = any(
                identity in frames
                or (
                    transcripts[identity]["start_us"] < packet["end_us"]
                    and transcripts[identity]["end_us"] > packet["start_us"]
                )
                for identity in block.evidence_ids
            )
            if not local:
                raise TaskError("模型只引用了相邻上下文，没有目标范围内的证据")
        from video_learner.notes.rendering import validate_body

        validate_body(block.body)
    if any(c in draft.title for c in "\r\n<>"):
        raise TaskError("模型章节标题包含无效结构")


def compose_chapter(
    book: Notebook,
    chapter: Chapter,
    config: Config,
    root: Path,
    provider: Provider,
) -> None:
    """调用供应商并校验草稿后更新章节，分配本地块 ID，再汇集具体疑点。"""
    packet, images = evidence_packet(book, chapter, config, root)
    draft = provider.compose(packet, images)
    validate_draft(draft, packet)
    chapter.title = draft.title
    chapter.blocks = []
    for index, block in enumerate(draft.blocks, 1):
        prefix = "fig" if block.kind == "figure" else "blk"
        identity = f"{prefix}-{chapter.id[3:]}-{index:03d}"
        chapter.blocks.append(NoteBlock(**block.model_dump(), id=identity, chapter_id=chapter.id))
    chapter.status = "completed"
    book.review.extend(chapter_review(chapter, draft.review))


def chapter_review(
    chapter: Chapter, reasons: list[str], target_id: str | None = None
) -> list[ReviewItem]:
    """汇集目标章节或单块的疑点，给正文疑点和模型提示登记可校验的修订目标。"""
    target_id = target_id or chapter.id
    items = []
    for block in chapter.blocks:
        if target_id != chapter.id and block.id != target_id:
            continue
        if block.category == "uncertain":
            items.append(
                ReviewItem(
                    reason="待核对内容：" + block.body[:300],
                    start_us=chapter.start_us,
                    end_us=chapter.end_us,
                    block_id=block.id,
                    evidence_ids=block.evidence_ids,
                )
            )
    for reason in reasons:
        items.append(
            ReviewItem(
                reason=reason,
                start_us=chapter.start_us,
                end_us=chapter.end_us,
                block_id=target_id,
            )
        )
    return items


def validate_notebook(book: Notebook, root: Path) -> None:
    """机械校验整份讲义的时间覆盖、唯一 ID、证据引用及缓存图片完整性。

    同时核对图片 PTS 与规范时间、文件哈希和可解码性。
    """
    if not 0 <= book.start_us < book.end_us <= book.source.duration_us:
        raise TaskError("文档范围越界")
    evidence = {}
    for segment in book.transcript:
        if not book.start_us <= segment.start_us < segment.end_us <= book.end_us:
            raise TaskError("转写证据时间越界")
        if segment.id in evidence:
            raise TaskError("证据 ID 重复")
        evidence[segment.id] = segment
    for frame in book.frames:
        if not book.start_us <= frame.at_us < book.end_us or frame.id in evidence:
            raise TaskError("图像证据时间越界或 ID 重复")
        if int(frame.pts * Fraction(frame.time_base) * US) - frame.origin_us != frame.at_us:
            raise TaskError("图像 PTS 与实际时间不一致")
        path = contained(root, frame.path)
        if not path.is_file():
            raise TaskError("图像证据资源缺失")
        if digest(path) != frame.sha256:
            raise TaskError("图像证据缓存内容已改变，请重新转换到独立目录")
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError) as exc:
            raise TaskError("图像证据缓存损坏") from exc
        evidence[frame.id] = frame
    expected_start = book.start_us
    identities = set()
    for chapter in book.chapters:
        if chapter.start_us != expected_start or chapter.end_us <= chapter.start_us:
            raise TaskError("章节覆盖存在空洞或重叠")
        expected_start = chapter.end_us
        if chapter.id in identities:
            raise TaskError("章节 ID 重复")
        identities.add(chapter.id)
        for block in chapter.blocks:
            if block.id in identities or block.chapter_id != chapter.id:
                raise TaskError("块 ID 重复或章节关系错误")
            identities.add(block.id)
            if block.category != "ai_addition" and not block.evidence_ids:
                raise TaskError("课程内容块缺少证据")
            if not set(block.evidence_ids) <= evidence.keys():
                raise TaskError("笔记引用不存在的证据")
            if block.kind == "figure" and (
                block.frame_id not in block.evidence_ids
                or not isinstance(evidence.get(block.frame_id), FrameEvidence)
            ):
                raise TaskError("图片块证据无效")
    if expected_start != book.end_us:
        raise TaskError("章节未覆盖完整范围")
    for item in book.review:
        if not book.start_us <= item.start_us < item.end_us <= book.end_us:
            raise TaskError("待核对项时间越界")
        if not set(item.evidence_ids) <= evidence.keys():
            raise TaskError("待核对项引用未知证据")
        if item.block_id is not None and item.block_id not in identities:
            raise TaskError("待核对项引用未知章节或块")
