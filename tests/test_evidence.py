import numpy as np
import pytest

from video_learner.common.config import Config
from video_learner.common.core import InputError, TaskError
from video_learner.media.evidence import audio_window, load_subtitles, sample_frames
from video_learner.media.io import extract_frame, inspect_source


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
    """取字幕时间交集时保留原时间，并按实际交集计算覆盖率；越界字幕不能静默导入。"""
    subtitle = tmp_path / "lesson.srt"
    subtitle.write_text("1\n00:20:00,000 --> 00:20:03,000\n原始字幕\n", encoding="utf-8")
    segments, report = load_subtitles(subtitle, 1_201_000_000, 1_204_000_000, 1_210_000_000)
    assert segments[0].start_us == 1_201_000_000
    assert segments[0].raw_start_us == 1_200_000_000
    assert report["coverage_ratio"] == pytest.approx(2 / 3)
    with pytest.raises(InputError):
        load_subtitles(subtitle, 0, 2_000_000, 2_000_000)


@pytest.mark.parametrize("end_us", [4_000_000, 3_980_000])
def test_sampling_tail_preserves_actual_frame_time(video, tmp_path, end_us):
    """最后采样点晚于片段内最后一帧时使用该帧；原请求、PTS 和精确换图语义仍可核对。"""
    source = inspect_source(video)
    frames = sample_frames(video, source, 1_950_000, end_us, Config(), tmp_path / "tail")
    assert [(f.requested_us, f.at_us) for f in frames] == [
        (1_950_000, 2_000_000),
        (3_950_000, 3_900_000),
    ]
    _, actual, pts = extract_frame(video, source, 3_900_000, end_us)
    assert (frames[-1].at_us, frames[-1].pts) == (actual, pts)
    with pytest.raises(TaskError, match="没有可用视频帧"):
        extract_frame(video, source, 3_950_000, end_us)
    with pytest.raises(TaskError, match="没有可用视频帧"):
        sample_frames(video, source, 3_950_000, end_us, Config(), tmp_path / "empty")
