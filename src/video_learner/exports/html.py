"""离线单文件 HTML 适配器：嵌入图片，MathML 公式，响应式阅读布局。"""

import base64
from html import escape
from pathlib import Path

from latex2mathml.converter import convert
from PIL import Image

from video_learner.common.storage import atomic_bytes
from video_learner.exports.document import ExportDocument, parser, safe_link

CSS = """
:root {color-scheme:light; font-family:system-ui,'Microsoft YaHei',sans-serif; color:#233047}
body {margin:0;background:#f1f4f8;line-height:1.85} main {max-width:900px;margin:32px auto;
padding:36px 52px;background:white;box-shadow:0 5px 25px #24334a12;border-radius:12px}
h1,h2,h3 {line-height:1.45;color:#163d65} h1 {font-size:2rem} h2 {margin-top:2.2em;
padding-bottom:.3em;border-bottom:1px solid #dce4ec} h3 {margin-top:1.8em}
a {color:#17649b;text-decoration:none} a:hover{text-decoration:underline}
img {max-width:100%;height:auto;display:block;margin:20px auto;border-radius:5px}
pre {padding:18px;background:#f3f6f9;overflow:auto;border:1px solid #dce4ec;border-radius:6px}
code {font-family:Consolas,monospace;font-size:.91em;overflow-wrap:anywhere}
table {border-collapse:collapse;width:100%;margin:20px 0;font-size:.94em}
th,td {border:1px solid #dce4ec;padding:8px 12px;text-align:left;overflow-wrap:anywhere}
th {background:#eaf1f7} blockquote {border-left:3px solid #7293b3;margin:18px 0;padding-left:18px}
.appendix {margin-top:60px;padding-top:12px;border-top:3px solid #dce4ec}
.math-block {overflow:auto;padding:14px 0} .math-source {background:#fff4d9;padding:8px}
@media(max-width:700px) {main{margin:0;padding:20px;border-radius:0}
table{display:block;overflow:auto}}
@media print {body{background:white} main{margin:0;box-shadow:none;padding:0}
img,pre{break-inside:avoid}}
"""


def write(document: ExportDocument, destination: Path) -> list[str]:
    """将共用节点渲染为静态 HTML；公式失败时显示原 LaTeX 并返回提示。"""
    md = parser()
    warnings = []

    def image(tokens, idx, options, env):
        """只嵌入预检的本地位图，不允许渲染器访问远程地址。"""
        token = tokens[idx]
        path = document.images[token.attrGet("src")]
        with Image.open(path) as source:
            mime = Image.MIME[source.format]
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'<img src="data:{mime};base64,{encoded}" alt="{escape(token.content, quote=True)}">'

    def link(tokens, idx, options, env):
        """构造安全链接，原始属性不会进入最终 HTML。"""
        href = escape(safe_link(tokens[idx].attrGet("href") or ""), quote=True)
        return f'<a href="{href}" rel="noreferrer">'

    def anchor(tokens, idx, options, env):
        return f'<a id="{escape(tokens[idx].attrGet("id"), quote=True)}"></a>'

    def math(tokens, idx, options, env):
        """MathML 由本地库生成；保留无法转换的完整公式源码。"""
        token = tokens[idx]
        block = token.type == "math_block"
        try:
            rendered = convert(token.content, display="block" if block else "inline")
        except Exception:
            warnings.append("HTML 中部分公式无法转换，已保留带标记的 LaTeX 原文。")
            rendered = f'<code class="math-source">LaTeX: {escape(token.content)}</code>'
        return f'<div class="math-block">{rendered}</div>' if block else rendered

    md.renderer.rules.update(
        image=image, link_open=link, anchor=anchor, math_inline=math, math_block=math
    )
    body = "\n".join(
        f'<section id="{part.name}" class="{"appendix" if part.name != "notes" else "lecture"}">'
        + md.renderer.render(part.tokens, md.options, {})
        + "</section>"
        for part in document.parts
    )
    page = (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
        "img-src data:; style-src 'unsafe-inline'; base-uri 'none'\">"
        f"<title>{escape(document.book.title)}</title><style>{CSS}</style></head>"
        f"<body><main>{body}</main></body></html>"
    )
    atomic_bytes(destination / "notes.html", page.encode("utf-8"))
    return list(dict.fromkeys(warnings))
