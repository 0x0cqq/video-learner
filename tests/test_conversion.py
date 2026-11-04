import json
from pathlib import Path

import pytest
from model_fakes import DeterministicGenerator, invoke

from video_learner.common.config import Config
from video_learner.common.core import InputError, TaskError
from video_learner.common.schemas import Notebook
from video_learner.notes.rendering import (
    AnchorConflict,
    image_dependencies,
    locate,
)
from video_learner.workflows.conversion import convert
from video_learner.workflows.replay import replay_conversion


def test_replay_reports_changed_input_or_model_draft(converted):
    """输入快照与模型草稿各自变化时指出所属边界，避免只比较最终文件。"""
    root, _ = converted
    input_path = root / ".work/composition-inputs/ch-001.json"
    original = input_path.read_bytes()
    frozen = json.loads(original)
    frozen["packet"]["title"] = "被改过的标题"
    input_path.write_text(json.dumps(frozen, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(TaskError, match="模型输入"):
        replay_conversion(root)
    input_path.write_bytes(original)

    # 冻结包版本随数据读取，既有 v2 记录无需迁移也能核对同一转换结果。
    legacy = json.loads(original)
    legacy["packet"]["packet_version"] = 2
    input_path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
    replay_conversion(root)
    input_path.write_bytes(original)

    draft_path = root / ".work/composition-drafts/ch-001.json"
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    draft["blocks"][0]["body"] = "与原生成结果不同的正文"
    draft_path.write_text(json.dumps(draft, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(TaskError, match="讲义索引"):
        replay_conversion(root)


def test_editorial_export_preserves_baseline_and_rejects_changed_evidence(converted, tmp_path):
    """人工审阅独立导出且零模型调用；原证据或基线手改变化时拒绝发布。"""
    import importlib.util

    from video_learner.common.storage import directory_lock

    path = Path(__file__).parents[1] / "tools/export_review.py"
    spec = importlib.util.spec_from_file_location("export_review", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root, video = converted
    original = (root / "notes.md").read_bytes()
    book = Notebook.model_validate_json((root / "notes.json").read_text(encoding="utf-8"))
    book.chapters[0].blocks[0].body = "经人工整理的说明。"
    edited = tmp_path / "edited.json"
    edited.write_text(book.model_dump_json(), encoding="utf-8")
    target = module.export_review(root, edited, tmp_path / "reviewed")
    assert (root / "notes.md").read_bytes() == original
    assert "经人工整理" in (target / "notes.md").read_text(encoding="utf-8")
    assert not (target / ".work/manifest.json").exists()
    for reference in image_dependencies((target / "notes.md").read_bytes()):
        assert (target / reference).is_file()
    with pytest.raises(InputError, match="原媒体目录"):
        module.export_review(root, edited, video.parent / "reviewed")
    with directory_lock(tmp_path / ".locked.lock"):
        with pytest.raises(InputError, match="正在写入"):
            module.export_review(root, edited, tmp_path / "locked")
    book.transcript[0].text = "改写原始转写"
    edited.write_text(book.model_dump_json(), encoding="utf-8")
    with pytest.raises(InputError, match="不得改变原始证据"):
        module.export_review(root, edited, tmp_path / "bad-evidence")
    (root / "notes.md").write_bytes(original + b"\nmanual edit\n")
    with pytest.raises(InputError, match="有手改"):
        module.export_review(root, edited, tmp_path / "bad-manual")


def test_force_cli_replaces_old_output(converted, monkeypatch):
    """显式 -f 删除整个旧输出并完成新转换；没有该参数时旧内容必须保留。"""

    from video_learner.common.core import InputError

    root, video = converted
    old = root / "manual.txt"
    old.write_text("旧手改", encoding="utf-8")
    with pytest.raises(InputError):
        convert(
            video,
            root,
            Config(),
            subtitle=video.with_suffix(".srt"),
            generator=DeterministicGenerator(),
        )
    assert old.read_text(encoding="utf-8") == "旧手改"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(
        "video_learner.workflows.conversion.create_generator",
        lambda config, events, **kwargs: DeterministicGenerator(),
    )
    result = invoke(
        "convert",
        video,
        output=root,
        subtitle=video.with_suffix(".srt"),
        force=True,
        no_review=True,
    )
    assert not old.exists()
    assert (root / "notes.md").is_file()
    assert (root / "notes.html").is_file() and (root / "notes.pdf").is_file()
    assert "notes.html" in result.stdout and "notes.pdf" in result.stdout


def test_force_preserves_output_when_preflight_or_lock_fails(converted):
    """参数错误或目标目录正在使用时，强制选项也不能提前删除旧结果。"""
    from video_learner.common.core import InputError
    from video_learner.common.storage import directory_lock

    root, video = converted
    old = (root / "notes.md").read_bytes()
    with pytest.raises(InputError):
        convert(
            video,
            root,
            Config(),
            end_us=0,
            force=True,
            subtitle=video.with_suffix(".srt"),
            generator=DeterministicGenerator(),
        )
    with directory_lock(root.parent / f".{root.name}.lock"), pytest.raises(InputError):
        convert(
            video,
            root,
            Config(),
            force=True,
            subtitle=video.with_suffix(".srt"),
            generator=DeterministicGenerator(),
        )
    assert (root / "notes.md").read_bytes() == old


def test_anchor_fences_and_conflicts(converted):
    """代码围栏内的锚点只是示例数据；真实重复锚点必须阻止定位。"""
    root, _ = converted
    data = (root / "notes.md").read_bytes()
    fake = b"```html\n<!-- vl:begin section fake -->\n```\n"
    assert locate(fake + data).keys() == locate(data).keys()
    with pytest.raises(AnchorConflict):
        locate(data + data)


@pytest.mark.parametrize(
    "body", ["```", "```python\nprint(1)", "- ```python\n  print(1)", "> ~~~\n> code"]
)
def test_unclosed_fences_fail_before_export(body):
    """不完整围栏应在模型正文校验阶段失败，避免消耗后续章节请求后才在导出时报错。"""
    from video_learner.notes.rendering import validate_body

    with pytest.raises(TaskError, match="代码围栏未闭合"):
        validate_body(body)
