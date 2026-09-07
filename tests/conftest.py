from fractions import Fraction

import av
import numpy as np
import pytest


@pytest.fixture
def video(tmp_path):
    """生成四秒、10 fps 的自造音视频：画面持续变化，1–3 秒静音，首尾为正弦音。

    明确设置音视频 PTS，供真实解码、重采样和静音区间对齐测试使用，无需私人素材。
    """
    folder = tmp_path / "input"
    folder.mkdir()
    path = folder / "lesson.mp4"
    with av.open(str(path), "w") as output:
        video = output.add_stream("mpeg4", rate=10)
        video.width, video.height, video.pix_fmt = 160, 96, "yuv420p"
        audio = output.add_stream("aac", rate=48000)
        audio.layout = "mono"
        for index in range(40):
            pixels = np.zeros((96, 160, 3), dtype=np.uint8)
            pixels[:, :, index // 10 % 3] = 80 + index * 3
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, 10)
            for packet in video.encode(frame):
                output.mux(packet)
            samples = np.zeros((1, 4800), dtype=np.float32)
            # 静音仍占据原时间线；后续波形比较可发现错误的静音压缩或时间重置。
            if index < 10 or index >= 30:
                samples[0] = 0.1 * np.sin(np.arange(4800) * 2 * np.pi * 440 / 48000)
            sound = av.AudioFrame.from_ndarray(samples, format="flt", layout="mono")
            sound.sample_rate = 48000
            sound.pts = index * 4800
            sound.time_base = Fraction(1, 48000)
            for packet in audio.encode(sound):
                output.mux(packet)
        for stream in (video, audio):
            for packet in stream.encode(None):
                output.mux(packet)
    return path
