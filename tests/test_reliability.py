import json
import os
import sys

import av
import numpy as np
import pytest
from test_conversion import DeterministicProvider
from test_conversion import converted as converted

from video_learner.cli import main
from video_learner.common.config import Config
from video_learner.common.core import InputError, TaskError, contained
from video_learner.common.schemas import Notebook
from video_learner.common.storage import write_json
from video_learner.media.evidence import audio_window
from video_learner.media.io import extract_frame, inspect_source
from video_learner.notes.composition import validate_notebook
from video_learner.workflows.conversion import convert
from video_learner.workflows.revision import revise


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_json_rejects_nonfinite_without_replacing_file(tmp_path, value):
    """非标准 JSON 数值必须在写入前失败，保留已有文件且不留下临时文件。"""
    path = tmp_path / "record.json"
    original = b'{"value": 1}\n'
    path.write_bytes(original)
    with pytest.raises(ValueError):
        write_json(path, {"value": value})
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_separate_audio_and_video_offsets_share_video_origin(video, tmp_path):
    """重新封装出不同音视频起点，确认两轨统一减去视频零点且音频延迟保留为静音。"""
    shifted = tmp_path / "shifted.mp4"
    with av.open(str(video)) as source, av.open(str(shifted), "w") as output:
        streams = {s.index: output.add_stream_from_template(s) for s in source.streams}
        for packet in source.demux():
            if packet.dts is None:
                continue
            seconds = 2 if packet.stream.type == "video" else 3
            shift = int(seconds / packet.time_base)
            packet.pts += shift
            packet.dts += shift
            packet.stream = streams[packet.stream.index]
            output.mux(packet)
    description = inspect_source(shifted)
    assert description.origin_us == 2_000_000
    # AAC 编码预留和封装编辑列表可能保留短前导，允许毫秒级差异，不要求恰好三秒。
    assert 2_970_000 <= description.tracks[1].start_us <= 3_010_000
    _, actual, _ = extract_frame(shifted, description, 50_000)
    assert actual == 100_000
    samples = audio_window(shifted, description, 0, 4_000_000)
    assert np.max(np.abs(samples[:12_000])) < 0.001
    assert np.max(np.abs(samples[20_000:28_000])) > 0.05


def test_cancelled_conversion_never_registers_success(video, tmp_path):
    """在证据已落盘后模拟用户中断，确认保留诊断但不登记版本或输出成功讲义。"""

    class Cancelled(DeterministicProvider):
        def compose(self, packet, images):
            raise KeyboardInterrupt

    subtitle = video.with_suffix(".srt")
    subtitle.write_text("1\n00:00:00,000 --> 00:00:04,000\n测试。\n", encoding="utf-8")
    output = tmp_path / "cancelled"
    with pytest.raises(KeyboardInterrupt):
        convert(video, output, Config(), subtitle=subtitle, provider=Cancelled())
    manifest = json.loads((output / ".work/manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "cancelled"
    assert not manifest["versions"]
    assert not (output / "notes.md").exists()
    assert (output / "transcript.jsonl").is_file()


def test_cli_cancellation_and_argument_exit_codes(monkeypatch, video):
    """直接调用 CLI 入口，验证中断和参数错误映射到约定退出码。"""

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("video_learner.cli.inspect_source", interrupt)
    monkeypatch.setattr(sys, "argv", ["video-learner", "inspect", str(video)])
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 130
    monkeypatch.setattr(sys, "argv", ["video-learner", "unknown-command"])
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 2


def test_cli_redirected_chinese_output_is_utf8(video):
    """在子进程强制继承旧代码页，确认实际重定向输出仍可按 UTF-8 解码。"""
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "video_learner.cli", "inspect", str(video)],
        env={**os.environ, "PYTHONIOENCODING": "cp936"},
        capture_output=True,
        check=True,
    )
    assert "媒体打开" in result.stderr.decode("utf-8")


def test_corrupt_cache_and_manifest_are_rejected(converted):
    """分别篡改图片缓存及清单顶层结构，确认修订不能复用损坏的本地证据。"""
    root, _ = converted
    book = Notebook.model_validate_json((root / "notes.json").read_text(encoding="utf-8"))
    image = contained(root, book.frames[0].path)
    image.write_bytes(image.read_bytes() + b"tampered")
    with pytest.raises(TaskError, match="缓存"):
        validate_notebook(book, root)
    manifest = root / ".work/manifest.json"
    manifest.write_text("[]", encoding="utf-8")
    with pytest.raises(InputError):
        revise(root, block="fig-001-002", at_us=1_000_000)


def test_changed_extraction_settings_are_not_silently_reused(converted, tmp_path):
    """修订时改变采样参数应触发指纹不一致，不能把旧证据当作新配置的提取结果。"""
    root, _ = converted
    config = tmp_path / "changed.toml"
    config.write_text("sample_seconds = 2\n", encoding="utf-8")
    with pytest.raises(InputError, match="提取配置"):
        revise(root, block="fig-001-002", at_us=1_000_000, config_path=config)
    assert not (root / "revisions/r002").exists()


def test_symlink_escape_is_rejected(tmp_path):
    """验证解析链接后的目录归属；Windows 无符号链接权限时用目录联接覆盖同类越界。"""
    root, other = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    other.mkdir()
    try:
        (root / "linked").symlink_to(other, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        import _winapi

        _winapi.CreateJunction(str(other), str(root / "linked"))
    with pytest.raises(InputError):
        contained(root, "linked/secret.png")
