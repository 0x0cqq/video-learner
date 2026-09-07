import hashlib
import json

import pytest
from typer.testing import CliRunner

from video_learner.cli import app
from video_learner.core import InputError, contained, output_path, parse_time, time_range
from video_learner.media import OffsetReader, crop_image, extract_frame, inspect_source


def test_time_and_paths(tmp_path):
    """覆盖微秒精度、非法时间和源/输出目录边界，防止解析与路径校验放宽契约。"""
    assert parse_time("00:20:00.123456") == 1_200_123_456
    for invalid in ("-1", "00:60:00", "NaN", "1.1234567"):
        with pytest.raises(InputError):
            parse_time(invalid)
    with pytest.raises(InputError):
        time_range(2, 1, 10)
    with pytest.raises(InputError):
        contained(tmp_path, "../secret")
    with pytest.raises(InputError):
        output_path(tmp_path, tmp_path / "output")


def test_probe_seek_and_readonly(video):
    """通过真实解码核对 seek 后实际帧时间、裁剪边界及素材哈希不变。"""
    before = hashlib.sha256(video.read_bytes()).hexdigest()
    source = inspect_source(video, decode=True)
    assert source.sampled_decode and not source.full_verified
    assert not source.diagnostics
    image, actual, pts = extract_frame(video, source, 1_250_000)
    assert image.size == (160, 96)
    assert actual == 1_300_000
    assert pts > 0
    assert crop_image(image, (10, 10, 30, 20)).size == (30, 20)
    with pytest.raises(InputError):
        crop_image(image, (150, 0, 30, 20))
    assert inspect_source(video, full=True).full_verified
    assert hashlib.sha256(video.read_bytes()).hexdigest() == before


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


def test_cli_json(video):
    result = CliRunner().invoke(app, ["inspect", str(video), "--json", "--decode"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["sampled_decode"] is True
