import numpy as np
import pytest

from video_learner.common.config import Config
from video_learner.common.core import InputError
from video_learner.common.storage import directory_lock
from video_learner.media.evidence import audio_window, load_subtitles, sample_frames
from video_learner.media.io import inspect_source


def test_audio_resampling_preserves_silence_and_clip_offset(video):
    """对比整段和带偏移片段的重采样波形，确认静音未被压缩、同一原视频时刻仍对应。"""
    source = inspect_source(video)
    full = audio_window(video, source, 0, 4_000_000)
    clipped = audio_window(video, source, 1_000_000, 4_000_000)
    assert len(full) == 64_000
    assert np.max(np.abs(full[24_000:40_000])) < 0.001
    assert np.max(np.abs(clipped[36_000:])) > 0.05
    assert np.allclose(full[20_000:60_000], clipped[4_000:44_000], atol=0.001)


def test_subtitles_clipping_coverage_and_invalid_times(tmp_path):
    """裁剪字幕时保留原时间，并按实际交集计算覆盖率；越界字幕不能静默导入。"""
    subtitle = tmp_path / "lesson.srt"
    subtitle.write_text("1\n00:20:00,000 --> 00:20:03,000\n原始字幕\n", encoding="utf-8")
    segments, report = load_subtitles(subtitle, 1_201_000_000, 1_204_000_000, 1_210_000_000)
    assert segments[0].start_us == 1_201_000_000
    assert segments[0].raw_start_us == 1_200_000_000
    assert report["coverage_ratio"] == pytest.approx(2 / 3)
    with pytest.raises(InputError):
        load_subtitles(subtitle, 0, 2_000_000, 2_000_000)


def test_sampling_and_os_lock(video, tmp_path):
    """请求落在两帧之间时应记录后一帧实际时间；目录锁须拒绝重入且退出后可重新获取。"""
    frames = sample_frames(
        video,
        inspect_source(video),
        1_250_000,
        4_000_000,
        Config(sample_seconds=1),
        tmp_path / "result",
    )
    assert frames[0].at_us == 1_300_000
    assert frames[0].requested_us == 1_250_000
    lock = tmp_path / "output.lock"
    with directory_lock(lock), pytest.raises(InputError):
        with directory_lock(lock):
            pass
    with directory_lock(lock):
        pass


def test_small_temporary_board_change_retains_candidate(video, tmp_path, monkeypatch):
    """重现细小符号被灰度均差忽略的情形，确认周期候选仍覆盖符号短暂出现的区间。"""
    from fractions import Fraction

    from PIL import Image

    description = inspect_source(video)
    description.duration_us = 90_000_000

    def board(path, source, at_us, end_us):
        """模拟只在 40–70 秒出现八像素符号的板书，并返回与原轨道 time base 一致的 PTS。"""
        image = Image.new("RGB", (160, 96), "darkgreen")
        if 40_000_000 <= at_us < 70_000_000:
            # 临时符号对全图均差几乎没有影响，用于覆盖板书细节被去重漏掉的退化情况。
            for x in range(80, 88):
                image.putpixel((x, 45), (255, 255, 255))
        pts = int(Fraction(at_us, 1_000_000) / Fraction(source.tracks[0].time_base))
        return image, at_us, pts

    monkeypatch.setattr("video_learner.media.evidence.extract_frame", board)
    frames = sample_frames(video, description, 0, 90_000_000, Config(), tmp_path / "board")
    assert any(40_000_000 <= f.at_us < 70_000_000 for f in frames)
