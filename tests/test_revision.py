import json

import pytest
from model_fakes import DeterministicGenerator

from video_learner.common.core import InputError
from video_learner.common.schemas import Draft, DraftBlock, Notebook
from video_learner.workflows.replay import replay_revision
from video_learner.workflows.revision import revise


def test_revision_preserves_nullable_baseline_settings(converted, tmp_path, monkeypatch):
    """自动语言空值与提取指纹保留；命令行供应商和指令覆盖配置文件。"""
    root, _ = converted
    config_path = tmp_path / "revision.toml"
    config_path.write_text(
        'provider = "deepseek"\ninstruction = "配置中的要求"\n', encoding="utf-8"
    )

    def factory(config, events, *, response_directory):
        """核对真正用于修订的配置，避免仅单测合并字典而遗漏加载基线时的空值丢失。"""
        assert config.asr_language is None
        assert config.provider == "qwen" and config.model == "qwen3.8-flash"
        assert config.api_key_env == "DASHSCOPE_API_KEY"
        assert config.instruction == "命令行要求"
        return DeterministicGenerator()

    monkeypatch.setattr("video_learner.workflows.revision.create_generator", factory)
    destination = revise(
        root,
        section="ch-001",
        instruction="命令行要求",
        config_path=config_path,
        model_provider="qwen",
    )
    assert (destination / "notes.md").is_file()
    assert (destination / "notes.html").is_file() and (destination / "notes.pdf").is_file()
    assert replay_revision(root, "r002").chapters[0].status == "completed"


def test_display_title_improvement_keeps_existing_evidence_revisable(converted, monkeypatch):
    """实际文件指纹不变时，标题展示改进不能把既有讲义误判成另一份素材。"""
    from video_learner.media.io import inspect_source

    root, _ = converted

    def renamed(path):
        """只改变探测器的展示标题，保留全部轨道和媒体身份。"""
        return inspect_source(path).model_copy(update={"title": "课程 · 课时"})

    monkeypatch.setattr("video_learner.workflows.versions.inspect_source", renamed)
    destination = revise(
        root, section="ch-001", instruction="压缩重复", generator=DeterministicGenerator()
    )
    assert (destination / "notes.md").is_file()


@pytest.mark.parametrize(("field", "value"), [("extraction_version", 3), ("schema_version", 1)])
def test_old_versions_are_rejected_without_writing(converted, field, value):
    """索引和提取旧版本均在读取或修改正文前拒绝，清单与原稿逐字保持。"""
    root, _ = converted
    path = root / ".work/manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest[field] = value
    path.write_text(json.dumps(manifest), encoding="utf-8")
    before = (root / "notes.md").read_bytes()
    with pytest.raises(InputError, match="重新转换到独立目录"):
        revise(root, block="fig-001-002", at_us=1_000_000)
    assert (root / "notes.md").read_bytes() == before
    assert json.loads(path.read_text(encoding="utf-8")) == manifest
    assert not (root / "revisions").exists()


def test_revision_conflicts_do_not_publish_versions(converted):
    """删除结束锚点制造定位冲突，确认只写诊断、不改原稿、不登记新版本。"""
    root, _ = converted
    markdown = root / "notes.md"
    markdown.write_bytes(markdown.read_bytes().replace(b"<!-- vl:end block blk-001-001 -->", b""))
    before = markdown.read_bytes()
    with pytest.raises(InputError, match="锚点"):
        revise(root, block="blk-001-001", instruction="改写", generator=DeterministicGenerator())
    assert markdown.read_bytes() == before
    assert list((root / ".work/conflicts").glob("*/suggestion.md"))
    assert not (root / "revisions/r002").exists()


def test_revision_concurrent_edit_source_mismatch(converted):
    """在供应商调用期间再次手改基线，再单独篡改素材，验证两类变化都会阻止发布。"""
    root, video = converted
    markdown = root / "notes.md"

    class ConcurrentProvider(DeterministicGenerator):
        def compose(self, packet, images):
            markdown.write_bytes(markdown.read_bytes() + "用户继续编辑\n".encode())
            return super().compose(packet, images)

    with pytest.raises(InputError, match="并发编辑"):
        revise(root, section="ch-001", instruction="重组", generator=ConcurrentProvider())
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
        root, base="r002", block="blk-001-001", instruction="核对正文", generator=Rewrite(True)
    )
    book = Notebook.model_validate_json((third / "notes.json").read_text(encoding="utf-8"))
    assert [item for item in book.review if item.block_id != "blk-001-001"] == baseline.review
    assert any(item.block_id == "blk-001-001" for item in book.review)
    fourth = revise(
        root, base="r003", section="ch-001", instruction="只保留正文", generator=Rewrite(False)
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
