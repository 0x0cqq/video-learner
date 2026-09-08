"""确定性 Markdown 渲染、严格锚点定位与可携带资源复制。"""

import html
import re
import shutil
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt
from PIL import Image

from video_learner.common.core import InputError, TaskError, contained, timestamp
from video_learner.common.schemas import Chapter, FrameEvidence, NoteBlock, Notebook
from video_learner.common.storage import atomic_bytes, write_json

MARKDOWN = MarkdownIt("commonmark", {"html": True})
ANCHOR = re.compile(rb"<!-- vl:(begin|end) (section|block) ([a-z][a-z0-9-]{1,63}) -->")
CATEGORIES = {"original": "原课整理", "ai_addition": "AI 补充解释", "uncertain": "待核对"}


class AnchorConflict(InputError):
    """无法确定安全替换范围。"""


@dataclass(frozen=True)
class Span:
    identity: str
    kind: str
    start: int
    end: int
    body_start: int
    body_end: int
    parent: str | None


def locate(data: bytes) -> dict[str, Span]:
    """扫描 UTF-8 Markdown 的受控锚点，返回包含起止标记的原始字节范围。

    忽略代码围栏和缩进代码里的示例锚点；重复、错位或未闭合结构直接报冲突，
    避免重新排版后定位而破坏 CRLF、手改文字及非目标字节。
    """
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AnchorConflict("当前 Markdown 不是 UTF-8，请先另存为 UTF-8") from exc
    offset = 0
    fence = None
    stack = []
    spans = {}
    seen = set()
    for line in data.splitlines(keepends=True):
        stripped = line.rstrip(b"\r\n")
        opening = re.match(rb"^ {0,3}(`{3,}|~{3,})(.*)$", stripped)
        if fence:
            if (
                opening
                and opening[1][:1] == fence[:1]
                and len(opening[1]) >= len(fence)
                and not opening[2].strip()
            ):
                fence = None
            offset += len(line)
            continue
        if opening:
            if opening[1].startswith(b"~") or b"`" not in opening[2]:
                fence = opening[1]
            offset += len(line)
            continue
        if stripped.startswith((b"    ", b"\t")):
            offset += len(line)
            continue
        match = ANCHOR.fullmatch(stripped)
        if match:
            operation, kind, identity = (v.decode("ascii") for v in match.groups())
            if operation == "begin":
                if identity in seen:
                    raise AnchorConflict(f"锚点 ID 重复：{identity}")
                if (kind == "section" and stack) or (
                    kind == "block" and (len(stack) != 1 or stack[0][0] != "section")
                ):
                    raise AnchorConflict(f"锚点嵌套结构冲突：{identity}")
                seen.add(identity)
                parent = stack[-1][1] if stack else None
                stack.append((kind, identity, offset, offset + len(line), parent))
            else:
                if not stack or stack[-1][:2] != (kind, identity):
                    raise AnchorConflict(f"锚点未正确闭合：{identity}")
                _, _, start, body_start, parent = stack.pop()
                spans[identity] = Span(
                    identity, kind, start, offset + len(line), body_start, offset, parent
                )
        elif b"<!-- vl:" in stripped:
            raise AnchorConflict("受控锚点格式已被修改；请恢复原锚点后重试")
        offset += len(line)
    if stack:
        raise AnchorConflict("锚点缺少结束标记，或被未闭合代码块遮蔽")
    return spans


def expected_spans(data: bytes, book: Notebook) -> dict[str, Span]:
    """将实际锚点与结构化索引逐项核对，拒绝缺失、新增、类型变化或跨章节移动。"""
    spans = locate(data)
    expected = {c.id for c in book.chapters}
    expected.update(b.id for c in book.chapters for b in c.blocks)
    if spans.keys() != expected:
        raise AnchorConflict("章节/段落锚点缺失或新增，不能安全定位；请核对生成快照")
    for chapter in book.chapters:
        if spans[chapter.id].kind != "section":
            raise AnchorConflict("章节锚点类型改变")
        for block in chapter.blocks:
            if spans[block.id].parent != chapter.id or spans[block.id].kind != "block":
                raise AnchorConflict("段落被移动到其他章节或类型改变")
    return spans


def validate_body(body: str) -> None:
    """限制模型正文为可安全嵌入的 Markdown，拒绝自造时间、受控结构和本地资源。

    同时检查代码围栏闭合，避免正文吞掉渲染器随后生成的锚点。
    """
    if re.search(r"\b\d{1,3}:\d{2}:\d{2}\b", body):
        raise TaskError("模型正文不得生成视频时间戳；请只提供证据 ID")
    for token in MARKDOWN.parse(body):
        if token.type == "fence":
            lines = body.splitlines()
            closing = lines[token.map[1] - 1].strip()
            if not re.fullmatch(
                re.escape(token.markup[0]) + "{" + str(len(token.markup)) + ",}", closing
            ):
                raise TaskError("模型正文的代码围栏未闭合")
        if token.type in ("html_block", "html_inline"):
            raise TaskError("模型正文不得包含 HTML 或锚点")
        for child in token.children or []:
            if child.type in ("image", "html_inline"):
                raise TaskError("模型正文不得生成图片路径或 HTML；请使用 figure 块")
            if child.type == "link_open":
                address = child.attrGet("href") or ""
                if urlsplit(address).scheme not in ("http", "https"):
                    raise TaskError("模型正文不得生成本地资源链接")
    if "<!-- vl:" in body:
        raise TaskError("模型正文包含受控锚点字符串")


def inline_text(value: str) -> str:
    """压平换行并转义行内 Markdown 控制字符，用于标题和疑点等非正文文本。"""
    value = " ".join(value.splitlines())
    return re.sub(r"([\\`*{}_\[\]<>])", r"\\\1", value)


def evidence_range(book: Notebook, identity: str) -> tuple[int, int]:
    """查询已登记证据的规范微秒范围；单张图片返回相同起止值表示一个时刻。"""
    for segment in book.transcript:
        if segment.id == identity:
            return segment.start_us, segment.end_us
    for frame in book.frames:
        if frame.id == identity:
            return frame.at_us, frame.at_us
    raise TaskError("引用未知证据")


def frame_of(book: Notebook, identity: str | None) -> FrameEvidence:
    for frame in book.frames:
        if frame.id == identity:
            return frame
    raise TaskError("图片证据缺失")


def render_block(block: NoteBlock, book: Notebook) -> bytes:
    """渲染正文与可选图注；修订标识保持隐藏，补充和疑点仍显式区分。"""
    parts = [f"<!-- vl:begin block {block.id} -->", ""]
    if block.category != "original":
        parts.extend([f"**{CATEGORIES[block.category]}**", ""])
    if block.kind == "figure":
        frame = frame_of(book, block.frame_id)
        parts.extend([f"![原视频截图 {timestamp(frame.at_us)}](assets/{frame.id}.png)", ""])
    if block.body.strip():
        parts.extend([block.body.strip(), ""])
    parts.extend([f"<!-- vl:end block {block.id} -->", ""])
    return ("\n".join(parts) + "\n").encode("utf-8")


def render_chapter(chapter: Chapter, book: Notebook) -> bytes:
    """渲染章节锚点、标题和正文；未完成章节显式提示，避免伪装成完整内容。"""
    heading = (
        f"<!-- vl:begin section {chapter.id} -->\n"
        f'<a id="{chapter.id}"></a>\n\n'
        f"## {inline_text(chapter.title)}\n\n"
    ).encode()
    if chapter.status != "completed":
        heading += "**本章未完成，请查看 review.md；不代表整段已转换。**\n\n".encode()
    return (
        heading
        + b"".join(render_block(b, book) for b in chapter.blocks)
        + (f"<!-- vl:end section {chapter.id} -->\n\n").encode()
    )


def render_notes(book: Notebook) -> bytes:
    """按结构化章节生成目录和整份初稿；保留手改的修订应传递原始 Markdown 字节。"""
    complete = all(c.status == "completed" for c in book.chapters)
    lines = [
        f"# {inline_text(book.title)}",
        "",
        *([] if complete else ["**部分章节未完成，详见 review.md。**", ""]),
        "[来源索引](sources.md) · [待核对事项](review.md)",
        "",
        "## 章节目录",
        "",
    ]
    for chapter in book.chapters:
        lines.append(f"- [{inline_text(chapter.title)}](#{chapter.id})")
    return ("\n".join(lines) + "\n\n").encode() + b"".join(
        render_chapter(chapter, book) for chapter in book.chapters
    )


def render_review(book: Notebook) -> bytes:
    """汇集未完成章节、模型疑点及手改未同步状态，生成独立核对文档。"""
    lines = [
        "# 来源与待核对",
        "",
        "证据 ID 和时间通过机械检查不代表内容正确。请核对公式、代码与关键步骤。",
        "",
    ]
    if book.sync_status == "manual_unverified":
        lines.extend(["当前文稿含保留的手改；结构化索引尚未核验同步，以 notes.md 为准。", ""])
    for chapter in book.chapters:
        if chapter.status != "completed":
            lines.append(
                f"- 未完成：{chapter.id}，{timestamp(chapter.start_us)}–{timestamp(chapter.end_us)}"
            )
    for item in book.review:
        lines.append(
            f"- {timestamp(item.start_us)}–{timestamp(item.end_us)}：{inline_text(item.reason)}"
            + (f"（{item.block_id}）" if item.block_id else "")
        )
    for chapter in book.chapters:
        for block in chapter.blocks:
            if block.sync_status == "manual_unverified":
                lines.append(f"- {block.id}：保留了用户手改，结构化正文与当前文稿未核验同步。")
    return ("\n".join(lines) + "\n").encode()


def render_sources(book: Notebook) -> bytes:
    """在阅读主线之外列出章节、块和真实证据时间，供定位原课及修订目标。"""
    lines = [
        "# 来源索引",
        "",
        "讲义中的隐藏锚点用于局部修订，目标 ID 见下表。完整关联保存在 notes.json。",
        "语音来源按实际音频切片区间记录，不表示句级对齐；截图时间为实际解码帧时间。",
        "公式需支持 LaTeX 的 Markdown 阅读器；代码未经本工具执行验证。",
        "",
    ]
    for chapter in book.chapters:
        lines.extend(
            [
                f"## {inline_text(chapter.title)} · {chapter.id}",
                "",
                f"原视频 {timestamp(chapter.start_us)}–{timestamp(chapter.end_us)}",
                "",
                "| 修订目标 | 类型 | 来源区间 | 截图时间 | 证据 ID |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for block in chapter.blocks:
            ranges = [evidence_range(book, identity) for identity in block.evidence_ids]
            interval = (
                f"{timestamp(min(r[0] for r in ranges))}–{timestamp(max(r[1] for r in ranges))}"
                if ranges
                else "—"
            )
            lines.append(
                f"| {block.id} | {CATEGORIES[block.category]} | {interval} | "
                + (
                    timestamp(frame_of(book, block.frame_id).at_us)
                    if block.kind == "figure"
                    else "—"
                )
                + " | "
                + ", ".join(block.evidence_ids)
                + " |"
            )
        lines.append("")
    return ("\n".join(lines) + "\n").encode()


class ImageHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources = []

    def handle_starttag(self, tag, attrs):
        """收集手改 HTML 图片的单一 src；拒绝无法完整复制依赖的 srcset。"""
        attributes = dict(attrs)
        if tag == "img":
            if attributes.get("srcset"):
                raise InputError("手改图片使用 srcset，无法确定全部依赖；请改用单一 src")
            if attributes.get("src"):
                self.sources.append(attributes["src"])


def image_dependencies(data: bytes) -> list[str]:
    """解析 Markdown 和 HTML 图片引用并按出现顺序去重，不把代码示例误作依赖。"""
    dependencies = []
    parser = ImageHTML()
    for token in MARKDOWN.parse(data.decode("utf-8")):
        if token.type == "html_block":
            parser.feed(token.content)
        for child in token.children or []:
            if child.type == "image":
                dependencies.append(child.attrGet("src") or "")
            elif child.type == "html_inline":
                parser.feed(child.content)
    return list(dict.fromkeys(dependencies + parser.sources))


def local_image(relative: str) -> str:
    """解码图片引用并拒绝远端地址、绝对路径和查询参数，返回本地路径部分。

    目录归属及上级跳转还需调用 contained 校验，此处不单独构成完整路径检查。
    """
    relative = html.unescape(unquote(relative))
    parsed = urlsplit(relative)
    if parsed.scheme or parsed.netloc or relative.startswith(("/", "\\")):
        raise InputError("可携带版本只接受版本目录内的本地图片；请将外部图片保存到 assets")
    if parsed.query:
        raise InputError("本地图片路径不能包含查询参数")
    return parsed.path


def copy_dependencies(data: bytes, base: Path, destination: Path) -> None:
    """验证并复制当前文稿的全部图片依赖，包括用户手加图片。

    已暂存的新图片优先；其余从基线复制，缺失、越界或非位图依赖均阻止发布。
    """
    for reference in image_dependencies(data):
        relative = local_image(reference)
        target = contained(destination, relative)
        # 新生成图片已暂存，不能再被基线旧图覆盖；现有目标也必须验证是有效图片。
        original = target if target.is_file() else contained(base, relative)
        if not original.is_file():
            raise InputError(f"图片依赖缺失：{relative}")
        try:
            with Image.open(original) as image:
                image.verify()
        except (OSError, ValueError) as exc:
            raise InputError("图片依赖损坏或不是支持的位图") from exc
        if original != target:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)


def stage_assets(
    book: Notebook, evidence_root: Path, destination: Path, base: Path | None = None
) -> None:
    """为讲义中选定的图块准备独立 assets，优先复用基线版本，再读取证据缓存。"""
    for chapter in book.chapters:
        for block in chapter.blocks:
            if block.kind == "figure":
                frame = frame_of(book, block.frame_id)
                relative = f"assets/{frame.id}.png"
                target = contained(destination, relative)
                if target.is_file():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                original = contained(base, relative) if base else None
                if original is None or not original.is_file():
                    original = contained(evidence_root, frame.path)
                shutil.copyfile(original, target)


def export_book(
    book: Notebook, evidence_root: Path, destination: Path, markdown: bytes | None = None
) -> None:
    """校验锚点及图片依赖后，将文稿、索引、来源和疑点写入目标目录。

    markdown 可传入保留手改的字节；这里只逐文件原子写入，整版发布和加锁由上层负责。
    """
    data = render_notes(book) if markdown is None else markdown
    expected_spans(data, book)
    stage_assets(book, evidence_root, destination)
    copy_dependencies(data, destination, destination)
    atomic_bytes(contained(destination, "notes.md"), data)
    atomic_bytes(contained(destination, "review.md"), render_review(book))
    atomic_bytes(contained(destination, "sources.md"), render_sources(book))
    write_json(contained(destination, "notes.json"), book.model_dump())
    write_json(contained(destination, "source.json"), book.source.model_dump())
