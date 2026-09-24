"""ReportLab 分页适配器；直接消费公共节点，嵌入中文字体与图片。"""

from html import escape
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from markdown_it.tree import SyntaxTreeNode
from PIL import Image as PILImage

from video_learner.common.core import InputError
from video_learner.exports.document import ExportDocument, safe_link


def font_path(explicit: Path | None = None) -> Path:
    """选择可嵌入的中文 TrueType 字体；其他平台可显式指定 TTF/TTC。"""
    candidates = (
        [explicit]
        if explicit
        else [
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/simsun.ttc"),
            Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        ]
    )
    for path in candidates:
        if path is not None and path.is_file():
            return path.resolve()
    raise InputError("PDF 需要中文 TrueType 字体，请用 --pdf-font 指定 TTF/TTC 文件")


def write(document: ExportDocument, destination: Path, *, font: Path | None = None) -> list[str]:
    """在临时目录渲染公式，完成或失败均清理中间图片。"""
    with TemporaryDirectory(prefix=".pdf-", dir=destination) as temporary:
        return render_pdf(document, destination, font, Path(temporary))


def render_pdf(
    document: ExportDocument, destination: Path, font: Path | None, temporary: Path
) -> list[str]:
    """排版正文、目录和附录；长表重复表头，长代码换行，超出公式子集时保留源码。"""
    try:
        from matplotlib.font_manager import FontProperties
        from matplotlib.mathtext import math_to_image
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import (
            Flowable,
            HRFlowable,
            Image,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:
        raise InputError("PDF 依赖未安装，请运行 uv sync --extra pdf") from exc

    path = font_path(font)
    try:
        face = TTFont("Lecture", str(path), subfontIndex=0)
        pdfmetrics.registerFont(face)
        bold = "Lecture"
        if path.name == "msyh.ttc" and path.with_name("msyhbd.ttc").is_file():
            pdfmetrics.registerFont(TTFont("LectureBold", str(path.with_name("msyhbd.ttc"))))
            bold = "LectureBold"
        pdfmetrics.registerFontFamily(
            "Lecture", normal="Lecture", bold=bold, italic="Lecture", boldItalic=bold
        )
    except Exception as exc:
        raise InputError("PDF 字体无法加载，请指定可嵌入的 TrueType TTF/TTC 字体") from exc
    width = A4[0] - 96
    body = ParagraphStyle(
        "body",
        fontName="Lecture",
        fontSize=10.5,
        leading=18,
        wordWrap="CJK",
        spaceAfter=8,
        splitLongWords=True,
    )
    styles = {"body": body}
    for level, size in ((1, 22), (2, 17), (3, 13), (4, 11), (5, 11), (6, 11)):
        styles[f"h{level}"] = ParagraphStyle(
            f"h{level}",
            parent=body,
            fontName=bold,
            fontSize=size,
            leading=size * 1.5,
            textColor=colors.HexColor("#163d65"),
            spaceBefore=14,
            spaceAfter=9,
            keepWithNext=True,
        )
    styles["code"] = ParagraphStyle(
        "code",
        parent=body,
        fontSize=8.5,
        leading=13,
        backColor=colors.HexColor("#f1f4f8"),
        borderPadding=8,
        spaceBefore=8,
        spaceAfter=12,
    )
    styles["cell"] = ParagraphStyle("cell", parent=body, fontSize=8.5, leading=13, spaceAfter=2)
    warnings = []

    class Bookmark(Flowable):
        """单独的章节锚点直接登记 PDF 目的地，不依赖空段落是否绘制。"""

        def __init__(self, name: str):
            super().__init__()
            self.name, self.width, self.height = name, 1, 1
            self.keepWithNext = True

        def draw(self):
            self.canv.bookmarkPage(self.name)

    trees = [(part.name, SyntaxTreeNode(part.tokens)) for part in document.parts]
    anchors = {name for name, _ in trees} | {
        n.attrGet("id") for _, tree in trees for n in tree.walk() if n.type == "anchor"
    }

    def text(value: str) -> str:
        """转义段落标记，标注字体覆盖不足，防止把正文当作排版指令。"""
        missing = {c for c in value if not c.isspace() and ord(c) not in face.face.charWidths}
        if missing:
            warnings.append("PDF 字体缺少字符：" + "".join(sorted(missing))[:30])
        return escape(value)

    formula_cache = {}

    def formula(value: str) -> tuple[Path, float, float] | None:
        """将 Mathtext 支持的公式栅格化；失败时由调用方展示完整 LaTeX。"""
        if value in formula_cache:
            return formula_cache[value]
        stream = BytesIO()
        try:
            math_to_image(
                "$" + value.strip() + "$",
                stream,
                prop=FontProperties(size=12, math_fontfamily="stix"),
                dpi=200,
                format="png",
            )
        except ValueError:
            warnings.append("PDF 中部分公式超出 Mathtext 支持范围，已保留带标记的 LaTeX 原文。")
            return None
        with PILImage.open(stream) as source:
            w, h = source.size
        image_path = temporary / f"formula-{len(formula_cache)}.png"
        image_path.write_bytes(stream.getvalue())
        formula_cache[value] = (image_path, w * 72 / 200, h * 72 / 200)
        return formula_cache[value]

    def inline(nodes: list[SyntaxTreeNode]) -> str:
        """将行内节点映射到受控 ReportLab 标记，不允许任意 HTML 或文件引用。"""
        parts = []
        for node in nodes:
            kind = node.type
            if kind in ("text", "code_inline"):
                parts.append(text(node.content))
            elif kind in ("softbreak", "hardbreak"):
                parts.append("<br/>" if kind == "hardbreak" else " ")
            elif kind in ("strong", "em", "s"):
                tag = {"strong": "b", "em": "i", "s": "strike"}[kind]
                parts.append(f"<{tag}>" + inline(node.children) + f"</{tag}>")
            elif kind == "link":
                href = safe_link(node.attrGet("href") or "")
                label = inline(node.children)
                if href and (not href.startswith("#") or href[1:] in anchors):
                    label = f'<a href="{escape(href, quote=True)}" color="#17649b">{label}</a>'
                parts.append(label)
            elif kind == "anchor":
                continue
            elif kind == "math_inline":
                rendered = formula(node.content)
                if rendered:
                    image_path, w, h = rendered
                    scale = min(1, 24 / max(h, 1), width / max(w, 1))
                    parts.append(
                        f'<img src="{escape(image_path.as_posix(), quote=True)}" '
                        f'width="{w * scale}" height="{h * scale}" valign="middle"/>'
                    )
                else:
                    parts.append("[LaTeX: " + text(node.content) + "]")
            elif kind != "image":
                parts.append(inline(node.children) if node.children else text(node.content))
        return "".join(parts)

    def picture(node: SyntaxTreeNode, available: float):
        """按页内可用面积等比缩放，保持全帧与可携带性。"""
        image = Image(str(document.images[node.attrGet("src")]))
        ratio = min(available / image.imageWidth, 440 / image.imageHeight, 1)
        image.drawWidth, image.drawHeight = image.imageWidth * ratio, image.imageHeight * ratio
        image.hAlign = "CENTER"
        image.spaceBefore, image.spaceAfter = 8, 12
        return image

    def flows(
        nodes: list[SyntaxTreeNode],
        depth: int = 0,
        bullet: str | None = None,
        cell: bool = False,
        available: float = width,
    ):
        """递归映射正文、列表和表格，分页交给 ReportLab，节点内容不重写。"""
        result = []
        for node in nodes:
            kind = node.type
            if kind == "anchor":
                result.append(Bookmark(node.attrGet("id")))
            elif kind in ("paragraph", "inline", "heading", "text"):
                result.extend(Bookmark(n.attrGet("id")) for n in node.walk() if n.type == "anchor")
                children = node.children if node.children else [node]
                if kind == "paragraph":
                    children = node.children[0].children
                style = (
                    styles[node.tag] if kind == "heading" else styles["cell" if cell else "body"]
                )
                style = ParagraphStyle(
                    "indented",
                    parent=style,
                    leftIndent=depth * 14,
                    bulletIndent=max(0, depth * 14 - 12),
                    bulletFontName="Lecture",
                )
                markup = inline(children)
                if markup:
                    result.append(Paragraph(markup, style, bulletText=bullet))
                    bullet = None
                for image in (n for n in node.walk() if n.type == "image"):
                    result.append(picture(image, available - depth * 14))
            elif kind == "image":
                result.append(picture(node, available - depth * 14))
            elif kind in ("bullet_list", "ordered_list"):
                start = int(node.attrGet("start") or 1)
                for number, item in enumerate(node.children, start):
                    label = f"{number}." if kind == "ordered_list" else "•"
                    result.extend(flows(item.children, depth + 1, label, cell, available))
            elif kind in ("fence", "code_block"):
                code = (
                    text(node.content.expandtabs(4)).replace(" ", "&#160;").replace("\n", "<br/>")
                )
                result.append(Paragraph(code, styles["code"]))
            elif kind == "math_block":
                rendered = formula(node.content)
                if rendered:
                    image_path, w, h = rendered
                    scale = min(1, available / max(w, 1), 440 / max(h, 1))
                    result.append(Image(str(image_path), width=w * scale, height=h * scale))
                    result.append(Spacer(1, 12))
                else:
                    result.append(
                        Paragraph("LaTeX 原文：<br/>" + text(node.content), styles["code"])
                    )
            elif kind == "table":
                rows = [n for n in node.walk() if n.type == "tr"]
                columns = max(len(row.children) for row in rows)
                col_width = available / columns
                data = [
                    [
                        flows(c.children, cell=True, available=col_width - 12) or ""
                        for c in row.children
                    ]
                    for row in rows
                ]
                table = Table(
                    data,
                    colWidths=[col_width] * columns,
                    repeatRows=1,
                    splitInRow=1,
                    hAlign="LEFT",
                    spaceBefore=8,
                    spaceAfter=12,
                )
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf1f7")),
                            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dce4ec")),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("TOPPADDING", (0, 0), (-1, -1), 6),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                        ]
                    )
                )
                result.append(table)
            elif kind == "hr":
                result.append(HRFlowable(width="100%", color=colors.HexColor("#dce4ec")))
            else:
                result.extend(
                    flows(node.children, depth + int(kind == "blockquote"), bullet, cell, available)
                )
        return result

    def footer(canvas, doc):
        """每页标注页码，长文采用简短页眉以保持定位。"""
        canvas.saveState()
        canvas.setFont("Lecture", 8)
        canvas.setFillColor(colors.HexColor("#718096"))
        canvas.drawRightString(A4[0] - 48, 27, str(doc.page))
        if doc.page > 1:
            canvas.drawString(48, A4[1] - 27, document.book.title[:40])
        canvas.restoreState()

    story = []
    for index, (name, tree) in enumerate(trees):
        if index:
            story.append(PageBreak())
        story.append(Bookmark(name))
        story.extend(flows(tree.children))
    # 显式管理句柄，排版失败时 Windows 也能清理暂存目录。
    with (destination / "notes.pdf").open("wb") as stream:
        pdf = SimpleDocTemplate(
            stream,
            pagesize=A4,
            leftMargin=48,
            rightMargin=48,
            topMargin=48,
            bottomMargin=48,
            title=document.book.title,
            author="Video Learner",
        )
        pdf.build(story, onFirstPage=footer, onLaterPages=footer)
    return list(dict.fromkeys(warnings))
