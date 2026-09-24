"""三个导出适配器共用的文档快照、语法节点和只读图片资源。"""

import re
import unicodedata
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

from markdown_it import MarkdownIt
from markdown_it.rules_inline import emphasis
from markdown_it.rules_inline.state_inline import StateInline
from markdown_it.token import Token
from mdit_py_plugins.dollarmath import dollarmath_plugin
from PIL import Image

from video_learner.common.core import InputError, contained
from video_learner.common.schemas import Notebook
from video_learner.notes.rendering import (
    expected_spans,
    image_dependencies,
    local_image,
    render_notes,
    render_review,
    render_sources,
)


def chinese_strong(state: StateInline, silent: bool) -> bool:
    """允许汉字与标点之间的双星号成对加粗，保留原解析器的嵌套及转义规则。"""
    start, first = state.pos, len(state.delimiters)
    if not emphasis.tokenize(state, silent):
        return False
    if state.src[start : state.pos] == "**":
        before = state.src[start - 1] if start else " "
        after = state.src[state.pos] if state.pos < state.posMax else " "
        # CommonMark 将汉字视为词内字符，导致「文**（词）**字」两侧均无法配对。
        opens = is_han(before) and unicodedata.category(after).startswith("P")
        closes = unicodedata.category(before).startswith("P") and is_han(after)
        for delimiter in state.delimiters[first:]:
            delimiter.open |= opens
            delimiter.close |= closes
    return True


def is_han(character: str) -> bool:
    """识别统一及兼容汉字（含扩展区），避免改变拉丁词内标点的强调规则。"""
    return unicodedata.name(character, "").startswith(
        ("CJK UNIFIED IDEOGRAPH-", "CJK COMPATIBILITY IDEOGRAPH-")
    )


def parser() -> MarkdownIt:
    """共享表格、公式与中文加粗语法；代码和转义字面量沿用 CommonMark。"""
    md = (
        MarkdownIt("commonmark", {"html": True})
        .enable(["table", "strikethrough"])
        .use(dollarmath_plugin, allow_labels=False)
    )
    md.inline.ruler.at("emphasis", chinese_strong)
    return md


class StaticHTML(HTMLParser):
    """手改 HTML 仅保留文字、图片和静态锚点，脚本等标签不进入导出器。"""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tokens: list[Token] = []

    def handle_starttag(self, tag, attrs):
        """资源仍由目录边界校验，HTML 属性不会原样传入阅读文档。"""
        attrs = dict(attrs)
        if tag == "img" and attrs.get("src"):
            self.tokens.append(
                Token(
                    "image",
                    "img",
                    0,
                    attrs={"src": local_image(attrs["src"]), "alt": attrs.get("alt", "")},
                    content=attrs.get("alt", ""),
                    children=[],
                )
            )
        elif tag == "a" and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", attrs.get("id", "")):
            self.tokens.append(Token("anchor", "", 0, attrs={"id": attrs["id"]}))

    def handle_data(self, data):
        """只保留字面文字，交由各适配器转义。"""
        self.tokens.append(Token("text", "", 0, content=data))


def static_tokens(tokens: list[Token]) -> list[Token]:
    """规范化手改 HTML 与图片引用；不执行素材代码或加载外部资源。"""
    result = []
    for token in tokens:
        if token.type in ("html_block", "html_inline"):
            html = StaticHTML()
            html.feed(token.content)
            result.extend(html.tokens)
            continue
        if token.children is not None:
            token.children = static_tokens(token.children)
        if token.type == "image":
            token.attrSet("src", local_image(token.attrGet("src") or ""))
        result.append(token)
    return result


def safe_link(value: str) -> str:
    """将随附索引映射到附录，只保留文内链接与普通网页、邮件链接。"""
    if value in ("sources.md", "review.md"):
        return "#" + value.removesuffix(".md")
    if value.startswith("#") or urlsplit(value).scheme.lower() in ("http", "https", "mailto"):
        return value
    return ""


@dataclass(frozen=True)
class DocumentPart:
    name: str
    markdown: bytes
    tokens: list[Token]


@dataclass(frozen=True)
class ExportDocument:
    book: Notebook
    parts: tuple[DocumentPart, ...]
    images: dict[str, Path]

    @property
    def markdown(self) -> bytes:
        return self.parts[0].markdown


def prepare_document(
    book: Notebook, evidence_root: Path, markdown: bytes | None = None, *, base: Path | None = None
) -> ExportDocument:
    """从讲义与当前手改文稿生成共享快照，预检图片；不写文件或调用模型。

    Markdown 字节用于无损导出，HTML/PDF 直接消费同一快照的语法节点。
    """
    data = render_notes(book) if markdown is None else markdown
    expected_spans(data, book)
    book = book.model_copy(deep=True)
    parts = tuple(
        DocumentPart(name, content, static_tokens(parser().parse(content.decode("utf-8"))))
        for name, content in (
            ("notes", data),
            ("sources", render_sources(book)),
            ("review", render_review(book)),
        )
    )
    frames = {f"assets/{frame.id}.png": frame.path for frame in book.frames}
    images = {}
    for reference in image_dependencies(data):
        relative = local_image(reference)
        original = contained(base, relative) if base is not None else None
        if original is None or not original.is_file():
            original = (
                contained(evidence_root, frames[relative]) if relative in frames else original
            )
        if original is None or not original.is_file():
            raise InputError(f"图片依赖缺失：{relative}")
        try:
            with Image.open(original) as image:
                image.verify()
        except (OSError, ValueError) as exc:
            raise InputError(f"图片损坏或格式不支持：{relative}") from exc
        images[relative] = original
    return ExportDocument(book, parts, images)
