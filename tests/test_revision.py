import json
import shutil

import pytest
from PIL import Image
from test_conversion import DeterministicProvider
from test_conversion import converted as converted

from video_learner.common.core import InputError
from video_learner.common.schemas import Draft, DraftBlock, Notebook
from video_learner.notes.rendering import image_dependencies, locate
from video_learner.workflows.replay import replay_revision
from video_learner.workflows.revision import revise


def test_revision_preserves_nullable_baseline_settings(video, tmp_path, monkeypatch):
    """自动语言的空值须保留；覆盖图文供应商和指令后仍能复用基线提取指纹。"""
    from video_learner.common.config import Config
    from video_learner.workflows.conversion import convert

    subtitle = video.with_suffix(".srt")
    subtitle.write_text("1\n00:00:00,000 --> 00:00:04,000\n测试\n", encoding="utf-8")
    root = convert(
        video,
        tmp_path / "nullable",
        Config(asr_language=None),
        subtitle=subtitle,
        provider=DeterministicProvider(),
    )
    config_path = tmp_path / "revision.toml"
    config_path.write_text(
        'provider = "deepseek"\ninstruction = "配置中的要求"\n', encoding="utf-8"
    )

    def factory(config, events):
        """核对真正用于修订的配置，避免仅单测合并字典而遗漏加载基线时的空值丢失。"""
        assert config.asr_language is None
        assert config.provider == "qwen" and config.model == "qwen3.8-flash"
        assert config.api_key_env == "DASHSCOPE_API_KEY"
        assert config.instruction == "命令行要求"
        return DeterministicProvider()

    monkeypatch.setattr("video_learner.workflows.revision.create_provider", factory)
    destination = revise(
        root,
        section="ch-001",
        instruction="命令行要求",
        config_path=config_path,
        model_provider="qwen",
    )
    assert (destination / "notes.md").is_file()
    assert replay_revision(root, "r002").chapters[0].status == "completed"


def test_display_title_improvement_keeps_existing_evidence_revisable(converted, monkeypatch):
    """实际文件指纹不变时，标题展示改进不能把既有讲义误判成另一份素材。"""
    from video_learner.media.io import inspect_source

    root, _ = converted

    def renamed(path):
        """只改变探测器的展示标题，保留全部轨道和媒体身份。"""
        return inspect_source(path).model_copy(update={"title": "课程 · 课时"})

    monkeypatch.setattr("video_learner.workflows.revision.inspect_source", renamed)
    destination = revise(
        root, section="ch-001", instruction="压缩重复", provider=DeterministicProvider()
    )
    assert (destination / "notes.md").is_file()


def test_old_extraction_version_is_rejected_without_writing(converted):
    """旧提取版本不自动迁移或发布修订，原稿和已登记的版本保持不变。"""
    root, _ = converted
    path = root / ".work/manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["extraction_version"] = 3
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
    assert replay_revision(root, "r002").sync_status == "manual_unverified"


def test_exact_image_revision_needs_no_provider_and_is_portable(converted, tmp_path, monkeypatch):
    """实际换帧保留图注手改及 CRLF；版本可携带、可继续修订，图片行手改则保留冲突。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    root, _ = converted
    markdown = root / "notes.md"
    before = (
        markdown.read_bytes()
        .replace(
            "变化后的原视频画面。".encode(),
            "手改图注必须保留。\n\n> 用户附注：请核对原视频。".encode(),
        )
        .replace(b"\n", b"\r\n")
    )
    markdown.write_bytes(before)
    destination = revise(root, block="fig-001-002", at_us=2_250_000)
    revised = (destination / "notes.md").read_bytes()
    assert b"00:00:02" in revised
    old_line = next(line for line in before.splitlines() if line.startswith(b"!["))
    new_line = next(line for line in revised.splitlines() if line.startswith(b"!["))
    assert new_line != old_line
    assert revised == before.replace(old_line, new_line)
    book = Notebook.model_validate_json((destination / "notes.json").read_text(encoding="utf-8"))
    frame = next(f for f in book.frames if f.id.startswith("frame-r002"))
    assert (frame.requested_us, frame.at_us) == (2_250_000, 2_300_000)
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
    manual_path = third / "notes.md"
    manual = manual_path.read_bytes().replace(b"![", b"![changed", 1)
    manual_path.write_bytes(manual)
    with pytest.raises(InputError, match="图片行已有手改"):
        revise(root, base="r003", block="fig-001-002", at_us=1_000_000)
    assert manual_path.read_bytes() == manual
    assert not (root / "revisions/r004").exists()


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
    assert replay_revision(root, "r002").chapters[0].blocks[1].frame_id == selected


def test_text_revisions_refresh_only_target_review_items(converted):
    """连续换图、单块和章节改写后清理旧疑点；范围外及字幕全局提示保留，新疑点有有效目标。"""
    root, _ = converted
    second = revise(root, block="fig-001-002", at_us=2_250_000)
    baseline = Notebook.model_validate_json((second / "notes.json").read_text(encoding="utf-8"))
    globals_before = [item for item in baseline.review if item.block_id is None]
    assert globals_before

    class Rewrite:
        def __init__(self, uncertain):
            self.uncertain = uncertain

        def compose(self, packet, images):
            """单块阶段返回正文疑点但空 review，章节阶段删除图片并提供新章节提示。"""
            return Draft(
                title="修订",
                blocks=[
                    DraftBlock(
                        kind="text",
                        body="待核对的正文" if self.uncertain else "已重新整理的正文",
                        category="uncertain" if self.uncertain else "original",
                        evidence_ids=[packet["transcript"][0]["id"]],
                        frame_id=None,
                    )
                ],
                review=[] if self.uncertain else ["新的章节疑点"],
            )

    third = revise(
        root, base="r002", block="blk-001-001", instruction="核对正文", provider=Rewrite(True)
    )
    book = Notebook.model_validate_json((third / "notes.json").read_text(encoding="utf-8"))
    assert [item for item in book.review if item.block_id != "blk-001-001"] == baseline.review
    assert any(item.block_id == "blk-001-001" for item in book.review)
    fourth = revise(
        root, base="r003", section="ch-001", instruction="只保留正文", provider=Rewrite(False)
    )
    final = Notebook.model_validate_json((fourth / "notes.json").read_text(encoding="utf-8"))
    assert [item for item in final.review if item.block_id is None] == globals_before
    scoped = [item for item in final.review if item.block_id is not None]
    assert [(item.block_id, item.reason) for item in scoped] == [("ch-001", "新的章节疑点")]
    assert "fig-001-002" not in (fourth / "review.md").read_text(encoding="utf-8")
    assert replay_revision(root, "r004").chapters[0].blocks[0].body == "已重新整理的正文"


def test_review_rejects_unknown_target(converted):
    """导出前拒绝悬空的疑点关联，防止成功版本的核对清单指向不存在的章节或块。"""
    from video_learner.common.core import TaskError
    from video_learner.notes.composition import validate_notebook

    root, _ = converted
    book = Notebook.model_validate_json((root / "notes.json").read_text(encoding="utf-8"))
    book.review[0].block_id = "deleted-block"
    with pytest.raises(TaskError, match="未知章节或块"):
        validate_notebook(book, root)
