import shutil
from pathlib import Path

import pytest
from PIL import Image
from test_conversion import converted as converted
from typer.testing import CliRunner

from video_learner.cli import app
from video_learner.common.core import InputError
from video_learner.common.storage import read_json
from video_learner.workflows.exporting import export


def pdf_font():
    """PDF 用例仅在已安装可选依赖与中文字体时运行，不下载字体。"""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    from video_learner.exports.pdf import font_path

    try:
        return font_path()
    except InputError:
        pytest.skip("本机没有测试所需的中文 TrueType 字体")


def test_export_portable_manual_markdown_and_static_html(converted, tmp_path):
    """移走视频和缓存后仍导出手改字节、表格、公式和图片，HTML 不执行素材脚本。"""
    root, video = converted
    portable = tmp_path / "portable"
    shutil.copytree(root, portable, ignore=shutil.ignore_patterns(".work", "usage.json"))
    video.unlink()
    Image.new("RGB", (32, 24), "red").save(portable / "assets/hand.png")
    path = portable / "notes.md"
    edited = (
        path.read_bytes().replace(b"\n", b"\r\n")
        + (
            '\r\n手改补充 **保留**。\r\n\r\n<img src="assets/hand.png" onerror="alert(1)">\r\n'
            "<script>alert(2)</script>\r\n\r\n| 项目 | 数值 |\r\n|---|---|\r\n| 甲 | 2 |\r\n"
            "\r\n公式 $x^2$。\r\n\r\n```html\r\n<script>literal()</script>\r\n```\r\n"
        ).encode()
    )
    path.write_bytes(edited)
    result = CliRunner().invoke(
        app,
        [
            "export",
            str(portable),
            "--output",
            str(tmp_path / "reading"),
            "--format",
            "markdown",
            "--format",
            "html",
        ],
    )
    assert result.exit_code == 0, result.output
    target = tmp_path / "reading"
    assert (target / "notes.md").read_bytes() == path.read_bytes() == edited
    assert (target / "assets/hand.png").read_bytes() == (portable / "assets/hand.png").read_bytes()
    html = (target / "notes.html").read_text(encoding="utf-8")
    assert "手改补充" in html and "<table>" in html and "<math" in html
    assert html.count('src="data:image/') == 2
    assert "<script>" not in html and "onerror=" not in html
    assert "&lt;script&gt;literal()&lt;/script&gt;" in html
    assert 'href="#sources"' in html and 'id="ch-001"' in html
    assert "手改" in (target / "sources.md").read_text(encoding="utf-8")
    assert read_json(target / "notes.json")["sync_status"] == "manual_unverified"
    copied = export(target, tmp_path / "re-export", ["html"])
    assert (copied / "notes.html").read_bytes() == (target / "notes.html").read_bytes()


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
    addition = table + code + r"公式 $\frac{a}{b}$ 和 $\notacommand{keep}$。"
    path.write_bytes(path.read_bytes() + addition.encode())
    target = export(root, tmp_path / "pdf", ["pdf"], pdf_font=font)
    reader = PdfReader(target / "notes.pdf")
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert len(reader.pages) > 5
    assert "自造声音与变化画面" in text and "79" in text
    assert "line_0" in text and "line_119" in text
    assert "notacommand" in text and "LaTeX" in text
    assert sum(len(page.images) for page in reader.pages) >= 2
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
    root, _ = converted
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


def test_export_explicit_version_and_invalid_format(converted, tmp_path):
    """显式基线可导出；无效格式在发布前给出明确错误。"""
    root, _ = converted
    target = export(root, tmp_path / "base", ["markdown"], base="r001")
    assert (target / "notes.md").read_bytes() == (root / "notes.md").read_bytes()
    with pytest.raises(InputError, match="导出格式"):
        export(root, tmp_path / "wrong", ["docx"])
