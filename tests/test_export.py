from pathlib import Path

import pytest
from markdown_it import MarkdownIt

from video_learner.common.core import InputError, TaskError
from video_learner.common.storage import read_json, write_json
from video_learner.exports.document import parser
from video_learner.workflows.exporting import export


def test_chinese_strong_keeps_literal_and_standard_markdown():
    """代码、转义、公式、URL、未闭合标记及普通英文强调不能被格式修复误改。"""
    source = (
        "代码 `按**（说明）**操作`，转义 按\\*\\*（说明）\\*\\*操作。\n\n"
        '```python\ntext = "按**（说明）**操作"\n```\n\n'
        '    text = "按**（说明）**操作"\n\n'
        "[链接](https://example.com/文**（说明）**字)\n\n"
        "未闭合**（说明）。 ** 空白 ** **尾空白 **\n\n"
        "foo**(bar)**baz **bold** *em* __strong__ a_b_c ***both***\n\n"
        "按*（说明）*操作 按***（说明）***操作\n\n"
        "实体 &#42;&#42;（说明）&#42;&#42; 字面量\n"
    )
    assert parser().render(source) == MarkdownIt("commonmark").render(source)
    tokens = parser().parseInline(r"$x**（y）**z$")[0].children
    assert [(token.type, token.content) for token in tokens] == [("math_inline", "x**（y）**z")]


def pdf_font():
    """PDF 用例使用常规依赖与本机中文字体，不下载字体。"""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    from video_learner.exports.pdf import font_path

    try:
        return font_path()
    except InputError:
        pytest.skip("本机没有测试所需的中文 TrueType 字体")


def test_pdf_paginates_tables_code_and_keeps_math_fallback(converted, tmp_path):
    """长表与代码跨页、中文可提取、公式与截图嵌入；超出公式子集时保留全文。"""
    font = pdf_font()
    from pypdf import PdfReader

    root, _ = converted
    path = root / "notes.md"
    table = "\n\n| 序号 | 说明 |\n|---|---|\n" + "\n".join(
        f"| {i} | 分页表格第 {i} 行 |" for i in range(80)
    )
    code = "\n\n```python\n" + "\n".join(f"print('line_{i}')" for i in range(120)) + "\n```\n"
    addition = "\n\n显示**（加粗）**文字。\n\n" + table + code
    addition += r"公式 $\frac{a}{b}$ 和 $\notacommand{keep}$。"
    addition += "\n\n### " + r"$\mathbb{Q}(\sqrt{2})$ 上的乘法与条件" * 8 + "\n\n"
    addition += r"对元素 $x+\sqrt{2}y$ 求逆时核对 $x^2-2y^2$ 的非零条件。" * 20
    addition += "\n\n推导：\n$$\nx^2\n$$\n这里使用有理数的封闭性 $x+y$。\n\n推导结束。\n"
    path.write_bytes(path.read_bytes() + addition.encode())
    target = export(root, tmp_path / "pdf", ["pdf"], pdf_font=font)
    reader = PdfReader(target / "notes.pdf")
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert len(reader.pages) > 5
    assert "自造声音与变化画面" in text and "79" in text
    assert "显示（加粗）文字" in text and "**" not in text
    assert "line_0" in text and "line_119" in text
    assert "推导结束" in text and text.count("非零条件") == 20
    assert "这里使用有理数的封闭性" in text
    assert "notacommand" in text and "LaTeX" in text
    assert sum(len(page.images) for page in reader.pages) >= 3
    assert text.count("序号") >= 2
    assert any("Mathtext" in warning for warning in read_json(target / "export.json")["warnings"])
    assert not list(target.glob(".pdf-*"))


@pytest.mark.parametrize(
    "reference", ["../private.png", "https://example.com/image.png", "assets/missing.png"]
)
def test_export_rejects_unavailable_or_external_images(converted, tmp_path, reference):
    """导出前阻止越界、网络和缺失图片，不发布半份 HTML。"""
    root, _ = converted
    path = root / "notes.md"
    path.write_bytes(path.read_bytes() + f"\n![手改图]({reference})\n".encode())
    target = tmp_path / "bad-export"
    with pytest.raises(InputError):
        export(root, target, ["html"])
    assert not target.exists()


def test_export_failure_preserves_source_and_publishes_nothing(converted, tmp_path):
    """后一个适配器失败也不留下已生成的 HTML，已有目录和源讲义保持不变。"""
    root, video = converted
    original = (root / "notes.md").read_bytes()
    target = tmp_path / "failed"
    with pytest.raises(InputError):
        export(root, target, ["html", "pdf"], pdf_font=Path("missing-font.ttf"))
    assert not target.exists() and not list(tmp_path.glob(".failed-*"))
    assert (root / "notes.md").read_bytes() == original
    with pytest.raises(InputError):
        export(root, root, ["html"])
    target.mkdir()
    (target / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(InputError):
        export(root, target, ["html"])
    assert (target / "keep.txt").read_text(encoding="utf-8") == "keep"
    video.unlink()
    with pytest.raises(InputError, match="原媒体目录"):
        export(root, video.parent / "reading", ["markdown"])


@pytest.mark.parametrize("damage", ["screenshot", "transcript", "coverage"])
def test_export_rejects_changed_evidence_and_invalid_structure(converted, tmp_path, damage):
    """有效位图替换、越界转写及章节空洞均不能借离线导出变成成功阅读副本。"""
    from PIL import Image

    root, _ = converted
    book = read_json(root / "notes.json")
    if damage == "screenshot":
        Image.new("RGB", (160, 96), "magenta").save(root / "assets/frame-000001.png")
    else:
        if damage == "transcript":
            book["transcript"][0]["end_us"] = book["end_us"] + 1_000_000
        else:
            book["chapters"][0]["start_us"] += 1
        write_json(root / "notes.json", book)
    target = tmp_path / "invalid-reading"
    with pytest.raises((InputError, TaskError)):
        export(root, target, ["markdown"])
    assert not target.exists()
    if damage == "screenshot":
        from model_fakes import DeterministicGenerator

        from video_learner.workflows.revision import revise

        with pytest.raises(InputError, match="登记截图"):
            revise(root, section="ch-001", instruction="重组", generator=DeterministicGenerator())
        assert not (root / "revisions/r002").exists()
        assert set(read_json(root / ".work/manifest.json")["versions"]) == {"r001"}


def test_image_edit_during_adapter_prevents_publication(converted, tmp_path, monkeypatch):
    """适配器工作期间图片被编辑时，三格式共用快照检查阻止发布混合资源。"""
    from PIL import Image

    from video_learner.exports import html

    root, _ = converted
    original_write = html.write

    def changed_image(document, destination):
        """模拟独立图片编辑器在 HTML 已消费图片后保存新画面。"""
        warnings = original_write(document, destination)
        Image.new("RGB", (160, 96), "magenta").save(next(iter(document.images.values())))
        return warnings

    monkeypatch.setattr(html, "write", changed_image)
    target = tmp_path / "concurrent-image"
    with pytest.raises(InputError, match="图片发生变化"):
        export(root, target, ["html", "markdown"])
    assert not target.exists()
