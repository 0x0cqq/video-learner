import json
import shutil

import pytest
from PIL import Image
from test_conversion import DeterministicProvider
from test_conversion import converted as converted

from video_learner.common.core import InputError
from video_learner.common.schemas import Draft, DraftBlock, Notebook
from video_learner.notes.rendering import image_dependencies, locate
from video_learner.workflows.revision import revise


def test_image_replacement_preserves_user_caption_and_extra_text(converted):
    """换图只修改受控图片行；手改图注与附加来源说明逐字节保留，手改图片行则冲突。"""
    from video_learner.notes.rendering import AnchorConflict, render_block
    from video_learner.workflows.revision import replace_figure

    root, _ = converted
    book = Notebook.model_validate_json((root / "notes.json").read_text(encoding="utf-8"))
    block = book.chapters[0].blocks[1]
    original = render_block(block, book)
    extra = "> 用户附注：请核对原视频。\r\n".encode()
    current = original.replace(block.body.encode(), "手改图注必须保留".encode()) + extra
    updated = replace_figure(current, original, block, book)
    assert updated == current
    with pytest.raises(AnchorConflict, match="已有手改"):
        replace_figure(current.replace(b"![", b"![changed", 1), original, block, book)


@pytest.mark.parametrize("version", [1, 2, 3])
def test_old_extraction_versions_are_rejected_without_writing(converted, version):
    """旧提取版本不自动迁移或发布修订，原稿和已登记的版本保持不变。"""
    root, _ = converted
    path = root / ".work/manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["extraction_version"] = version
    path.write_text(json.dumps(manifest), encoding="utf-8")
    before = (root / "notes.md").read_bytes()
    with pytest.raises(InputError, match="版本不兼容"):
        revise(root, block="fig-001-002", at_us=1_000_000)
    assert (root / "notes.md").read_bytes() == before
    assert json.loads(path.read_text(encoding="utf-8")) == manifest
    assert not (root / "revisions").exists()


def test_old_schema_is_rejected_before_loading_index(converted):
    """索引结构变更后通过清单版本给出可操作错误，不触碰旧数据或读取旧字段。"""
    root, _ = converted
    path = root / ".work/manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 1
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(InputError, match="重新转换到独立目录"):
        revise(root, block="fig-001-002", at_us=1_000_000)
    assert json.loads(path.read_text(encoding="utf-8")) == manifest
    assert not (root / "revisions").exists()


def test_text_revision_preserves_current_bytes_and_user_assets(converted):
    """给基线加入目标手改、CRLF 和自绘图，验证模型能看到手改且非目标字节及依赖保留。"""
    root, _ = converted
    markdown = root / "notes.md"
    data = markdown.read_bytes().replace("自造声音与变化画面。".encode(), "手改目标正文".encode())
    data = data.replace(b"\n", b"\r\n") + "\r\n用户结尾 ![手绘图](assets/hand.png)\r\n".encode()
    markdown.write_bytes(data)
    Image.new("RGB", (10, 10), "red").save(root / "assets/hand.png")
    before = locate(data)["blk-001-001"]

    class Rewrite:
        def compose(self, packet, images):
            """断言修订输入含当前手改，再返回有真实引用的单块替换，供字节保护检查。"""
            assert "手改目标正文" in packet["current_markdown"]
            return Draft(
                title="修订",
                blocks=[
                    DraftBlock(
                        kind="text",
                        body="按要求修改后的正文",
                        category="original",
                        evidence_ids=[packet["transcript"][0]["id"]],
                        frame_id=None,
                    )
                ],
                review=[],
            )

    destination = revise(root, block="blk-001-001", instruction="压缩表述", provider=Rewrite())
    revised = (destination / "notes.md").read_bytes()
    after = locate(revised)["blk-001-001"]
    assert revised[: after.start] == data[: before.start]
    assert revised[after.end :] == data[before.end :]
    assert markdown.read_bytes() == data
    assert (destination / "assets/hand.png").read_bytes() == (root / "assets/hand.png").read_bytes()
    assert destination.name == "r002"
    assert (
        json.loads((destination / "notes.json").read_text(encoding="utf-8"))["sync_status"]
        == "manual_unverified"
    )


def test_exact_image_revision_needs_no_provider_and_is_portable(converted, tmp_path, monkeypatch):
    """无模型凭据执行换帧，再复制版本并以其为新基线换帧，核对完整画面和版本关系。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    root, _ = converted
    before = (root / "notes.md").read_bytes()
    destination = revise(root, block="fig-001-002", at_us=2_250_000)
    revised = (destination / "notes.md").read_bytes()
    assert b"00:00:02" in revised
    book = Notebook.model_validate_json((destination / "notes.json").read_text(encoding="utf-8"))
    frame = next(f for f in book.frames if f.id.startswith("frame-r002"))
    assert frame.at_us == 2_300_000
    with Image.open(root / frame.path) as image:
        assert image.size == (160, 96)
    assert (root / "notes.md").read_bytes() == before
    portable = tmp_path / "portable"
    shutil.copytree(destination, portable)
    for reference in image_dependencies((portable / "notes.md").read_bytes()):
        assert (portable / reference).is_file()
    third = revise(root, base="r002", block="fig-001-002", at_us=1_500_000)
    assert third.name == "r003"
    manifest = json.loads((root / ".work/manifest.json").read_text(encoding="utf-8"))
    assert manifest["versions"]["r003"]["base"] == "r002"


def test_revision_conflicts_do_not_publish_versions(converted):
    """删除结束锚点制造定位冲突，确认只写诊断、不改原稿、不登记新版本。"""
    root, _ = converted
    markdown = root / "notes.md"
    markdown.write_bytes(markdown.read_bytes().replace(b"<!-- vl:end block blk-001-001 -->", b""))
    before = markdown.read_bytes()
    with pytest.raises(InputError, match="锚点"):
        revise(root, block="blk-001-001", instruction="改写", provider=DeterministicProvider())
    assert markdown.read_bytes() == before
    assert list((root / ".work/conflicts").glob("*/suggestion.md"))
    assert not (root / "revisions/r002").exists()


def test_revision_concurrent_edit_source_mismatch(converted):
    """在供应商调用期间再次手改基线，再单独篡改素材，验证两类变化都会阻止发布。"""
    root, video = converted
    markdown = root / "notes.md"

    class ConcurrentProvider(DeterministicProvider):
        def compose(self, packet, images):
            markdown.write_bytes(markdown.read_bytes() + "用户继续编辑\n".encode())
            return super().compose(packet, images)

    with pytest.raises(InputError, match="并发编辑"):
        revise(root, section="ch-001", instruction="重组", provider=ConcurrentProvider())
    assert "用户继续编辑" in markdown.read_text(encoding="utf-8")
    assert not (root / "revisions/r002").exists()
    video.write_bytes(video.read_bytes() + b"changed")
    with pytest.raises(InputError, match="源素材"):
        revise(root, block="fig-001-002", at_us=1_000_000)


def test_revise_rejects_missing_or_escaping_manual_images(converted):
    """把缺失或越界图片加入手改文稿，确认依赖预检阻止生成不可携带版本。"""
    root, _ = converted
    markdown = root / "notes.md"
    original = markdown.read_bytes()
    for path in ("assets/missing.png", "../secret.png"):
        markdown.write_bytes(original + f"\n![手改]({path})\n".encode())
        with pytest.raises(InputError):
            revise(root, block="fig-001-002", at_us=1_000_000)
    assert not (root / "revisions/r002").exists()


def test_caption_revision_keeps_the_existing_frame(converted):
    """替身核对只收到原选中图片，确保改图注不会隐式触发自然语言重新选图。"""
    root, _ = converted
    original = json.loads((root / "notes.json").read_text(encoding="utf-8"))
    selected = original["chapters"][0]["blocks"][1]["frame_id"]

    class Caption:
        def compose(self, packet, images):
            """同时核对模型可见清单和实际传图只有原图，避免仅修改清单却泄漏其他候选。"""
            assert [identity for identity, _ in images] == [selected]
            assert [frame["id"] for frame in packet["frames"]] == [selected]
            return Draft(
                title="图注",
                blocks=[
                    DraftBlock(
                        kind="figure",
                        body="精简后的图注",
                        category="original",
                        evidence_ids=[selected],
                        frame_id=selected,
                    )
                ],
                review=[],
            )

    destination = revise(root, block="fig-001-002", instruction="缩短图注", provider=Caption())
    revised = json.loads((destination / "notes.json").read_text(encoding="utf-8"))
    assert revised["chapters"][0]["blocks"][1]["frame_id"] == selected


def test_revision_can_switch_multimodal_provider_without_reextracting(converted, monkeypatch):
    """捕获修订工厂配置，验证切换 Qwen 可复用既有证据且目标章节外字节保持不变。"""
    root, _ = converted
    before = (root / "notes.md").read_bytes()
    selected = []

    def factory(config, events):
        selected.append(config)
        return DeterministicProvider()

    monkeypatch.setattr("video_learner.workflows.revision.create_provider", factory)
    destination = revise(root, section="ch-001", instruction="重新整理", model_provider="qwen")
    assert selected[0].provider == "qwen"
    assert selected[0].model == "qwen3.8-flash"
    assert selected[0].api_key_env == "DASHSCOPE_API_KEY"
    after = (destination / "notes.md").read_bytes()
    old, new = locate(before)["ch-001"], locate(after)["ch-001"]
    assert before[: old.start] == after[: new.start]
    assert before[old.end :] == after[new.end :]
    assert (root / "notes.md").read_bytes() == before
