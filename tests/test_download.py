"""以回环 HTTP 服务验证实际 yt-dlp 下载，站点解析替身不访问公网。"""

import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import av
import pytest
from model_fakes import DeterministicGenerator, invoke
from yt_dlp import YoutubeDL
from yt_dlp.extractor.common import InfoExtractor
from yt_dlp.utils import DownloadError

from video_learner.common.core import InputError
from video_learner.common.schemas import Notebook
from video_learner.common.storage import read_json
from video_learner.workflows.ingestion import resolve_source


@pytest.fixture
def media_server(video):
    """提供自造合流和分离音视频；HTTP 下载和 PyAV 解码均执行真实代码。"""
    for kind, name in (("video", "picture.mp4"), ("audio", "sound.m4a")):
        with av.open(str(video)) as source, av.open(str(video.with_name(name)), "w") as output:
            stream = next(item for item in source.streams if item.type == kind)
            target = output.add_stream_from_template(stream)
            for packet in source.demux(stream):
                if packet.dts is not None:
                    packet.stream = target
                    output.mux(packet)

    class Handler(SimpleHTTPRequestHandler):
        """仅供回环测试；不将请求 URL 写到测试输出。"""

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(video.parent)))
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


@pytest.mark.parametrize("split", [False, True], ids=["combined", "separate"])
def test_url_cli_download_convert_revise_export(media_server, video, tmp_path, monkeypatch, split):
    """验证三入口共享业务：真实下载、时间证据、三格式生成、离线换图及素材身份核验。"""
    urls = []
    if split:

        class LessonIE(InfoExtractor):
            """只替换 Bilibili 站点解析，保留分 P；后续选择、下载和检查使用真正 yt-dlp。"""

            _VALID_URL = r"https://www\.bilibili\.com/video/(?P<id>BV[\w]+)"

            def _real_extract(self, url):
                urls.append(url)
                return {
                    "id": "BV1bK411W797_p2",
                    "title": "测试课程 · 第二 P",
                    "formats": [
                        {
                            "format_id": "1",
                            "url": media_server + "/sound.m4a?token=private-signature",
                            "vcodec": "none",
                            "acodec": "aac",
                        },
                        {
                            "format_id": "2",
                            "url": media_server + "/picture.mp4?token=private-signature",
                            "vcodec": "mpeg4",
                            "acodec": "none",
                        },
                    ],
                }

        def downloader(options):
            """使用自造解析器优先处理测试站点，不替换 yt-dlp 的格式选择和下载实现。"""
            client = YoutubeDL(options, auto_init=False)
            client.add_info_extractor(LessonIE())
            client.add_default_info_extractors()
            return client

        monkeypatch.setattr("video_learner.media.download.YoutubeDL", downloader)
        url = "https://www.bilibili.com/video/BV1bK411W797?p=2"
    else:
        url = media_server + "/lesson.mp4?token=private-signature"
    downloads = tmp_path / "downloads"
    checked = invoke("inspect", url, download_dir=downloads, json=True, full=True)
    inspected = json.loads(checked.stdout)
    assert inspected["adapter"] == "yt-dlp" and inspected["full_verified"]
    downloaded = next(path for path in downloads.iterdir() if path.is_dir())
    # 已下载目录再次作为本地输入时保留 yt-dlp 身份，且无需再次访问站点。
    invoke("inspect", downloaded, json=True, decode=True)
    if split:
        assert inspected["title"] == "测试课程 · 第二 P"
        assert inspected["bvid"] == "BV1bK411W797" and urls == [url]
        assert {item["file"] for item in inspected["tracks"]} == {"audio.m4a", "video.mp4"}
    else:
        assert len({item["file"] for item in inspected["tracks"]}) == 1

    subtitle = video.with_suffix(".srt")
    subtitle.write_text("1\n00:00:00,000 --> 00:00:04,000\n自造音视频内容。\n", encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(
        "video_learner.workflows.conversion.create_generator",
        lambda *args, **kwargs: DeterministicGenerator(),
    )
    # 不传 --download-dir 时，URL 自动使用默认目录，不需要启用开关。
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "output"
    converted = invoke(
        "convert", url, output=root, start="1", end="3.5", subtitle=subtitle, no_review=True
    )
    book = Notebook.model_validate(read_json(root / "notes.json"))
    assert book.source.adapter == "yt-dlp" and book.start_us == 1_000_000
    assert all(1_000_000 <= frame.at_us < 3_500_000 for frame in book.frames)
    for extension in ("md", "html", "pdf"):
        assert (root / f"notes.{extension}").is_file()
    source_path = Path(read_json(root / ".work/source-local.json")["path"])
    assert source_path.parent == tmp_path / ".cache/downloads"
    original = (root / "notes.md").read_bytes()
    invoke("revise", root, base="r001", block="fig-001-002", frame="2")
    assert (root / "revisions/r002/notes.pdf").is_file()
    assert (root / "notes.md").read_bytes() == original
    invoke("export", root, base="r002", output=tmp_path / "reading")
    for folder in (downloads, tmp_path / ".cache/downloads", root):
        for file in folder.rglob("*"):
            if file.suffix in (".json", ".jsonl", ".md", ".html"):
                assert "private-signature" not in file.read_text(encoding="utf-8")
    assert "private-signature" not in checked.output + converted.output
    # 元数据与媒体一起参与基线指纹，不能通过篡改下载标题偷偷改变素材身份。
    (source_path / "download.json").write_text('{"title":"已篡改"}', encoding="utf-8")
    invoke("revise", root, base="r001", block="fig-001-002", frame="2", code=2)


@pytest.mark.parametrize("failure", ["playlist", "live", "download", "cancel", "invalid_media"])
def test_download_failure_never_publishes(tmp_path, monkeypatch, failure):
    """集合/直播提前拒绝，下载错误、中断及坏媒体清理临时数据，不暴露签名 URL。"""

    class FailedDownload:
        """外部故障发生在下载阶段前后，帮助定位失败发布与清理边界。"""

        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download):
            if failure == "playlist":
                return {"_type": "playlist", "entries": []}
            return {"id": "test", "title": "测试", "is_live": failure == "live"}

        def process_ie_result(self, info, download):
            destination = Path(self.options["outtmpl"]).parent
            (destination / "media.mp4").write_bytes(b"partial invalid media")
            if failure == "cancel":
                raise KeyboardInterrupt
            if failure == "download":
                raise DownloadError("https://example.test/video?token=private-signature")

    monkeypatch.setattr("video_learner.media.download.YoutubeDL", FailedDownload)
    target = tmp_path / "downloads"
    result = invoke(
        "inspect",
        "https://example.test/video",
        download_dir=target,
        code=130
        if failure == "cancel"
        else 2
        if failure in ("playlist", "live", "invalid_media")
        else 1,
    )
    assert list(target.iterdir()) == []
    assert "private-signature" not in result.output


def test_download_output_protection_and_bvid(tmp_path, monkeypatch):
    """冲突目录在联网前拒绝，force 保留旧文件；裸 BVID 与路径共用明确的输入分派。"""
    calls = []
    monkeypatch.setattr(
        "video_learner.workflows.ingestion.download_source",
        lambda *args: calls.append(args) or tmp_path,
    )
    downloads = tmp_path / "downloads"
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "manual.txt"
    marker.write_text("已有手改", encoding="utf-8")
    for target, directory, force in (
        (output, downloads, False),
        (output, output / "downloads", True),
        (downloads / "output", downloads, True),
        (Path.cwd(), downloads, True),
    ):
        with pytest.raises(InputError):
            resolve_source("https://example.test/video", directory, output=target, force=force)
    assert not calls and marker.read_text(encoding="utf-8") == "已有手改"
    assert resolve_source("BV1bK411W797", downloads) == tmp_path
    assert calls[0][0] == "https://www.bilibili.com/video/BV1bK411W797"
    assert resolve_source(tmp_path / "local.mp4") == tmp_path / "local.mp4"
    for value in ("file:///secret", "https://user:password@example.test/video", "https://"):
        with pytest.raises(InputError):
            resolve_source(value, downloads)
