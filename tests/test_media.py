import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from video_learner.cli import app
from video_learner.common.core import InputError, output_path
from video_learner.media.io import OffsetReader, inspect_source


def test_prefix_is_verified_not_suffix(video, tmp_path):
    """对比有前缀、无前缀、坏头和截断文件，证明不能只因扩展名为 m4s 就跳过九字节。"""
    prefixed = tmp_path / "prefixed.m4s"
    prefixed.write_bytes(b"0" * 9 + video.read_bytes())
    assert inspect_source(prefixed, decode=True).tracks[0].prefix_bytes == 9
    with OffsetReader(prefixed) as reader:
        assert reader.read(4) == video.read_bytes()[:4]
        reader.seek(-5, 2)
        assert reader.read() == video.read_bytes()[-5:]
    bad = tmp_path / "bad.m4s"
    bad.write_bytes(b"0" * 9 + b"bad header")
    with pytest.raises(InputError):
        inspect_source(bad)
    normal = tmp_path / "normal.m4s"
    normal.write_bytes(video.read_bytes())
    assert inspect_source(normal).tracks[0].prefix_bytes == 0
    truncated = tmp_path / "truncated.mp4"
    truncated.write_bytes(video.read_bytes()[:100])
    with pytest.raises(InputError):
        inspect_source(truncated)


@pytest.mark.parametrize("failure", ["corrupt", "unreadable"])
def test_inspect_reports_partial_media_failure(video, monkeypatch, failure):
    """候选媒体内容损坏或头部不可读都归一为逐文件诊断，JSON 仍保留有效视频信息。"""
    video.with_name("broken.aac").write_bytes(b"synthetic invalid audio")
    if failure == "unreadable":
        original_open = Path.open

        def denied(path, *args, **kwargs):
            """模拟音频被权限或文件锁阻止读取，其他实际媒体仍正常处理。"""
            if path.name == "broken.aac":
                raise PermissionError("synthetic read denial")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", denied)
    result = CliRunner().invoke(app, ["inspect", str(video.parent), "--json"])
    assert result.exit_code == 1
    source = json.loads(result.stdout)
    assert source["opened"] is False
    assert any("broken.aac" in message for message in source["diagnostics"])
    assert source["tracks"]


def test_metadata_failure_does_not_mean_media_open_failure(video):
    """元数据解析错误独立报告，不应使已成功打开的媒体状态变为失败。"""
    video.with_name("videoInfo.json").write_text("invalid JSON", encoding="utf-8")
    source = inspect_source(video.parent)
    assert source.opened is True
    assert source.diagnostics


def test_force_output_keeps_source_and_working_directory_protected(video):
    """强制覆盖仍拒绝源目录、工作目录及文件路径，避免递归删除重要输入。"""
    from pathlib import Path

    for target in (video.parent, video.parent.parent, Path.cwd(), Path.cwd().parent, video):
        with pytest.raises(InputError):
            output_path(video, target, force=True)
