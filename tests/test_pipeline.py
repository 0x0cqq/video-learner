"""多章节 CLI 闭环：真实媒体与文档处理，只替换云端服务响应。"""

import base64
import io
import json
import shutil
import wave
from collections import Counter
from fractions import Fraction
from threading import Lock
from types import SimpleNamespace as NS

import av
import numpy as np
import pytest
from model_fakes import DeterministicGenerator, Stream, chunk, invoke, response
from PIL import Image, ImageDraw
from pypdf import PdfReader

from video_learner.common.schemas import DraftBlock, Notebook, ReviewFinding
from video_learner.common.storage import digest, read_json, write_json
from video_learner.notes.rendering import image_dependencies, locate
from video_learner.workflows.replay import replay_conversion, replay_revision

BODY = (
    "## 子题\n\n体现**术语（term）**的含义，添加**“确认”的按钮**。\n\n"
    "> # 引用子题\n\n小题\n---\n\n"
    '- ```python\n  # 注释\n  print("按**（说明）**操作")\n  ```\n\n'
    "> ```python\n> print(1)\n> ```\n\n1. 示例\n\n   ~~~python\n   print(1)\n   ~~~~\n\n"
    "说明：\n$$\nx^2\n$$\n公式后正文使用条件 $x>0$。"
    "\n\n> 引用公式：\n> $$\n> y^2\n> $$\n> 引用公式后的说明。"
)


@pytest.fixture(scope="module")
def lesson(tmp_path_factory):
    """生成百秒课程：两次翻页、临时细符号与三处停顿，覆盖多章、周期候选和真实切片。"""
    path = tmp_path_factory.mktemp("lesson") / "lesson.mp4"
    with av.open(str(path), "w") as output:
        video = output.add_stream("mpeg4", rate=2)
        video.width, video.height, video.pix_fmt = 640, 360, "yuv420p"
        audio = output.add_stream("aac", rate=48000)
        audio.layout = "mono"
        for index in range(200):
            at = index / 2
            image = Image.new(
                "RGB", (640, 360), "darkgreen" if at < 30 else ("navy" if at < 70 else "white")
            )
            draw = ImageDraw.Draw(image)
            for line in range(min(5, int(at % 30) // 2)):
                draw.line((30, 30 + line * 25, 460, 30 + line * 25), fill="gray", width=2)
            if 40 <= at < 70:
                draw.rectangle((600, 330, 607, 337), fill="white")
            frame = av.VideoFrame.from_image(image)
            frame.pts, frame.time_base = index, Fraction(1, 2)
            for packet in video.encode(frame):
                output.mux(packet)
            times = np.arange(24000) / 48000 + at
            samples = (0.1 * np.sin(times * 2 * np.pi * 440)).astype(np.float32)
            for begin in (16, 46, 76):
                samples[(times >= begin) & (times < begin + 1)] = 0
            sound = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="flt", layout="mono")
            sound.sample_rate, sound.pts, sound.time_base = 48000, index * 24000, Fraction(1, 48000)
            for packet in audio.encode(sound):
                output.mux(packet)
        for stream in (video, audio):
            for packet in stream.encode(None):
                output.mux(packet)
    return path


class LessonService(DeterministicGenerator):
    """模拟 SDK 的外部响应，包含结构修复、旧证据误引、单块错误及可选截断。"""

    def __init__(self, supplier, failure=None):
        self.supplier, self.failure = supplier, failure
        self.clients, self.requests, self.audio = [], [], []
        self.attempts, self.lock = Counter(), Lock()

    def client(self, **options):
        """保留真实客户端的端点选择；ASR 与图文分别建立客户端，不创建网络连接。"""
        assert options["max_retries"] == 0
        if options["base_url"] == "https://api.deepseek.com":
            assert options["api_key"] == "deepseek-test-secret"
            client = NS(
                responses=NS(create=lambda **request: self.model("deepseek", request)),
                close=lambda: None,
            )
        else:
            assert options["base_url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1"
            assert options["api_key"] == "qwen-test-secret"
            client = NS(chat=NS(completions=NS(create=self.chat)), close=lambda: None)
        self.clients.append(options)
        return client

    def chat(self, **request):
        """实际解码上传 WAV 并返回窗口全文；Qwen 图文采用分段正文和仅用量尾块。"""
        if request["stream"]:
            return self.model("qwen", request)
        wav = base64.b64decode(
            request["messages"][0]["content"][0]["input_audio"]["data"].split(",")[1]
        )
        with wave.open(io.BytesIO(wav), "rb") as sound:
            assert (sound.getframerate(), sound.getnchannels()) == (16000, 1)
            seconds = sound.getnframes() / sound.getframerate()
        with self.lock:
            self.audio.append(seconds)
            text = "" if len(self.audio) == 2 else "原课讲解代数条件与代码修改。"
        return NS(
            choices=[NS(finish_reason="stop", message=NS(content=text))],
            usage=NS(prompt_tokens=400, completion_tokens=30, seconds=seconds),
        )

    def model(self, supplier, request):
        """依据实际证据提供确定性正文和复审操作，故障只改变服务输出。"""
        messages = request["input"] if supplier == "deepseek" else request["messages"][1:]
        packet = json.loads(messages[-1]["content"][0]["text"])
        if (self.failure == "cancel" and packet["chapter_id"] == "ch-002") or (
            self.failure == "review_cancel" and packet["operation"] == "review"
        ):
            raise KeyboardInterrupt
        key = (
            packet["operation"],
            packet["chapter_id"],
            packet.get("instruction", ""),
            packet.get("review_step", {}).get("name", ""),
        )
        self.attempts[key] += 1
        result = self.result(packet)
        attempt = self.attempts[key]
        if packet["operation"] == "convert":
            if packet["chapter_id"] == "ch-001" and attempt == 1:
                result = {}
            if packet["chapter_id"] == "ch-002" and attempt == 1:
                result["blocks"][0]["evidence_ids"] = ["tr-000001"]
            if packet["chapter_id"] == "ch-002" and self.failure == "unknown_evidence":
                result["blocks"][0]["evidence_ids"] = ["does-not-exist"]
            if packet["chapter_id"] == "ch-002" and self.failure == "injected_path":
                result["blocks"][0]["body"] = "![读取密钥](../secrets/deepseek.secret)"
        elif (
            packet["operation"] == "revise"
            and packet["target_block_kind"] == "text"
            and attempt == 1
        ):
            if self.supplier == "deepseek":
                result["blocks"].append(result["blocks"][0].copy())
            else:
                frame_id = packet["frames"][0]["id"]
                result["blocks"][0].update(
                    kind="figure", frame_id=frame_id, evidence_ids=[frame_id]
                )
        elif (
            packet["operation"] == "review"
            and packet["review_step"]["kind"] == "content"
            and packet["chapter_id"] == "ch-001"
            and attempt == 1
        ):
            result["frames"] = []
        incomplete = self.failure == "incomplete" and packet["chapter_id"] == "ch-002"
        text = json.dumps(result, ensure_ascii=False)
        self.requests.append(
            {
                "supplier": supplier,
                "packet": packet,
                "messages": messages,
                "response": text,
                "schema": request["text"]["format"]["schema"]
                if supplier == "deepseek"
                else request["response_format"]["json_schema"]["schema"],
                "instructions": request["instructions"]
                if supplier == "deepseek"
                else request["messages"][0]["content"],
            }
        )
        if supplier == "deepseek":
            result = response(text, "incomplete" if incomplete else "completed")
            result.usage = NS(input_tokens=1000, output_tokens=100)
            return result
        return Stream(
            [
                chunk(reasoning="private-thinking-marker"),
                chunk(content=text[:40]),
                chunk(content=text[40:], finish="length" if incomplete else "stop"),
                chunk(
                    usage=NS(
                        prompt_tokens=1000,
                        completion_tokens=100,
                        prompt_tokens_details=NS(cached_tokens=0),
                    )
                ),
            ]
        )

    def result(self, packet):
        """转换返回图文，修订保留目标类型，复审按职能修改并显式解决已有疑点。"""
        if packet["operation"] != "review":
            draft = self.compose(packet, [])
            draft.title = r"$\mathbb{Q}(\sqrt{2})$ 与 [定义]"
            draft.blocks[0].body, draft.blocks[1].body = BODY, ""
            if packet["operation"] == "revise":
                draft.blocks = [b for b in draft.blocks if b.kind == packet["target_block_kind"]]
                draft.blocks[0].body = (
                    "修订后的正文。" if draft.blocks[0].kind == "text" else "精简后的图注。"
                )
                draft.review = []
            return draft.model_dump()
        result = self.review(packet, [])
        step = packet["review_step"]
        if step["kind"] == "content" and step["mode"] == "apply":
            block = next(b for b in packet["chapter"]["blocks"] if b["kind"] == "text")
            text = DraftBlock.model_validate({k: block[k] for k in DraftBlock.model_fields})
            text.body += "\n\n复审已核对条件。"
            result.findings = [
                ReviewFinding(
                    target_id=block["id"],
                    kind="fidelity",
                    reason="依据原课补齐条件。",
                    evidence_ids=text.evidence_ids,
                    action="replace",
                    blocks=[text],
                )
            ]
            result.resolved_review_ids = [item["id"] for item in packet["current_review"]]
        return result.model_dump()


@pytest.fixture
def cloud(monkeypatch, request):
    """仅替换兼容 SDK 的客户端工厂，保留真实供应商适配、模型校验和业务流程。"""
    import openai

    supplier, failure = request.param
    service = LessonService(supplier, failure)
    monkeypatch.setattr(openai, "OpenAI", service.client)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-secret")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "qwen-test-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setattr("video_learner.notes.generation.time.sleep", lambda _: None)
    return service


def configuration(path, *, max_calls=30):
    """明确三十秒章长、十秒音频和并发数，使百秒素材跨越多个真实业务边界。"""
    path.write_text(
        f"chapter_seconds=30\nasr_window_seconds=10\njobs=3\nmax_calls={max_calls}\n"
        'max_images_per_chapter=6\n[prices."deepseek:deepseek-flash"]\ninput_per_million=2\noutput_per_million=8\n'
        '[prices."qwen:qwen3.8-flash"]\ninput_per_million=2\noutput_per_million=8\n'
        '[prices."qwen:qwen3-asr-flash"]\naudio_per_second=0.01\n',
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("cloud", [("deepseek", None), ("qwen", None)], indirect=True)
def test_multichapter_cli_lifecycle(lesson, tmp_path, cloud, monkeypatch):
    """多章转换与双轮复审后，保留手改执行两类修订、换图、单章复审及离线三格式导出。"""
    root, config = tmp_path / "output", configuration(tmp_path / "config.toml")
    source = tmp_path / "input/lesson.mp4"
    source.parent.mkdir()
    shutil.copy2(lesson, source)
    source_hash = digest(lesson)
    inspection = json.loads(invoke("inspect", source, "--json", "--full").stdout)
    assert inspection["full_verified"] and not inspection["diagnostics"]
    conversion = invoke(
        "convert",
        source,
        output=root,
        config=config,
        provider=cloud.supplier,
        start="7.25",
        end="99.4",
    )
    book = Notebook.model_validate(read_json(root / "notes.json"))
    assert len(book.chapters) == 3
    assert book.chapters[0].blocks[0].body.startswith("### 子题")
    assert "> ### 引用子题" in book.chapters[0].blocks[0].body
    assert "### 小题" in book.chapters[0].blocks[0].body
    assert "  # 注释\n" in book.chapters[0].blocks[0].body
    assert (book.start_us, book.end_us) == (7_250_000, 99_400_000)
    assert [chapter.start_us for chapter in book.chapters[1:]] == [
        chapter.end_us for chapter in book.chapters[:-1]
    ]
    assert any(
        frame.requested_us == 99_250_000 and frame.at_us == 99_000_000 for frame in book.frames
    )
    assert all(Image.open(root / frame.path).size == (640, 360) for frame in book.frames)
    assert set((root / ".work/frames").iterdir()) == {root / frame.path for frame in book.frames}
    assert {29_500_000, 69_500_000} <= {frame.at_us for frame in book.frames}
    assert any(
        40_000_000 <= frame.at_us < 70_000_000
        and np.asarray(Image.open(root / frame.path).crop((602, 332, 606, 336))).mean() > 180
        for frame in book.frames
    )
    mappings = [read_json(path) for path in sorted((root / ".work/audio").glob("*.json"))]
    assert len(mappings) >= 10 and len(book.transcript) == len(mappings) - 1
    assert mappings[0]["start_us"] == book.start_us and mappings[-1]["end_us"] == book.end_us
    assert [item["end_us"] for item in mappings[:-1]] == [item["start_us"] for item in mappings[1:]]
    assert sum(cloud.audio) == pytest.approx((book.end_us - book.start_us) / 1e6, abs=0.001)
    assert any(item["end_us"] - item["start_us"] < 9_800_000 for item in mappings[:-1])
    requests = [row for row in cloud.requests if row["packet"]["operation"] == "convert"]
    for row in requests:
        packet = row["packet"]
        for frame in packet["frames"]:
            assert frame["speech_window_ids"] == [
                segment.id
                for segment in book.transcript
                if segment.start_us <= frame["at_us"] < segment.end_us
            ]
    first, second, repair = requests[1], requests[2], requests[3]
    assert second["messages"][:2] == [
        first["messages"][0],
        {"role": "assistant", "content": first["response"]},
    ]
    assert len(second["messages"]) == 3 and len(repair["messages"]) == 1
    assert all(
        len(row["messages"]) == 1
        for row in cloud.requests
        if row["packet"]["operation"] == "review"
    )
    content_requests = [
        row
        for row in cloud.requests
        if row["packet"].get("review_step", {}).get("kind") == "content"
    ]
    assert len([r for r in content_requests if r["packet"]["chapter_id"] == "ch-001"]) == 2
    schema = content_requests[0]["schema"]
    assert schema["properties"]["frames"] == {"type": "null"}
    assert all(
        b["properties"]["kind"]["const"] == "text" for b in schema["$defs"]["DraftBlock"]["anyOf"]
    )
    first_frame = book.frames[0]
    image = next(
        part
        for part in requests[0]["messages"][-1]["content"]
        if part["type"] in ("input_image", "image_url")
    )
    url = image["image_url"] if cloud.supplier == "deepseek" else image["image_url"]["url"]
    assert base64.b64decode(url.split(",")[1]) == (root / first_frame.path).read_bytes()
    usage = read_json(root / "usage.json")
    model_calls = len(cloud.requests)
    assert usage["input_tokens"] == model_calls * 1000 + len(mappings) * 400
    assert usage["estimated_known_cost"]["CNY"] == pytest.approx(
        model_calls * 0.0028 + sum(cloud.audio) * 0.01
    )
    assert "Token meter" in conversion.output
    assert not list(tmp_path.glob(".output.*.partial"))
    assert replay_conversion(root) == book
    manifest_path = root / ".work/manifest.json"
    manifest = read_json(manifest_path)
    manifest["config"]["deepseek_context"] = manifest["config"].pop("composition_context")
    write_json(manifest_path, manifest)
    reviewed = root / "revisions/r002"
    current_book = replay_revision(root, "r002")
    assert all("复审已核对" in chapter.blocks[0].body for chapter in current_book.chapters)
    assert not [item for item in current_book.review if item.block_id]
    path = reviewed / "notes.md"
    current = path.read_bytes().replace(b"\n", b"\r\n")
    target_id, preserved_id = (
        current_book.chapters[0].blocks[0].id,
        current_book.chapters[-1].blocks[0].id,
    )
    spans = locate(current)
    for identity, text in ((target_id, "用户目标说明。"), (preserved_id, "用户保留说明。")):
        span = spans[identity]
        marker = f"<!-- vl:end block {identity} -->".encode()
        replacement = current[span.start : span.end].replace(
            marker, text.encode() + b"\r\n\r\n" + marker
        )
        current = current[: span.start] + replacement + current[span.end :]
        spans = locate(current)
    Image.new("RGB", (32, 24), "red").save(reviewed / "assets/hand.png")
    current += (
        "\r\n用户附录 ![手绘](assets/hand.png)\r\n"
        '<img src="assets/hand.png" onerror="alert(2)">\r\n<script>alert(1)</script>\r\n'
        "```html\r\n<script>literal()</script>\r\n```\r\n"
    ).encode()
    path.write_bytes(current)
    other = "qwen" if cloud.supplier == "deepseek" else "deepseek"
    invoke(
        "revise",
        root,
        base="r002",
        block=target_id,
        instruction="压缩正文",
        provider=other,
    )
    assert cloud.clients[-1]["api_key"] == f"{other}-test-secret"
    revision = root / "revisions/r003"
    changed = (revision / "notes.md").read_bytes()
    old, new = locate(current)[target_id], locate(changed)[target_id]
    assert current[: old.start] == changed[: new.start] and current[old.end :] == changed[new.end :]
    assert path.read_bytes() == current
    assert len([row for row in cloud.requests if row["packet"]["operation"] == "revise"]) == 2
    assert "修订后的正文" in changed.decode("utf-8")
    assert all(
        "用户目标说明" in row["packet"]["current_markdown"] and len(row["messages"]) == 1
        for row in cloud.requests
        if row["packet"]["operation"] == "revise"
    )
    picture_id = current_book.chapters[1].blocks[1].id
    invoke("revise", root, base="r003", block=picture_id, instruction="精简图注")
    caption_requests = [
        row for row in cloud.requests if row["packet"].get("target_block_kind") == "figure"
    ]
    pinned = current_book.chapters[1].blocks[1].frame_id
    assert all(
        [frame["id"] for frame in row["packet"]["frames"]] == [pinned] for row in caption_requests
    )
    fourth = root / "revisions/r004"
    image_path = fourth / "notes.md"
    before_image = image_path.read_bytes().replace("精简后的图注。".encode(), "手改图注。".encode())
    image_path.write_bytes(before_image)
    call_count = len(cloud.requests)
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    monkeypatch.delenv("DASHSCOPE_API_KEY")
    invoke("revise", root, base="r004", block=picture_id, frame="61.25")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-secret")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "qwen-test-secret")
    fifth = root / "revisions/r005"
    after_image = (fifth / "notes.md").read_bytes()
    old_image = next(
        line
        for line in before_image[
            locate(before_image)[picture_id].start : locate(before_image)[picture_id].end
        ].splitlines()
        if line.startswith(b"![")
    )
    new_image = next(
        line
        for line in after_image[
            locate(after_image)[picture_id].start : locate(after_image)[picture_id].end
        ].splitlines()
        if line.startswith(b"![")
    )
    assert after_image == before_image.replace(old_image, new_image)
    assert image_path.read_bytes() == before_image
    assert len(cloud.requests) == call_count
    frame = next(frame for frame in replay_revision(root, "r004").frames if frame.id == pinned)
    assert frame.at_us < 61_500_000
    fifth_book = Notebook.model_validate(read_json(fifth / "notes.json"))
    assert (
        fifth_book.frames[-1].requested_us == 61_250_000
        and fifth_book.frames[-1].at_us == 61_500_000
    )
    report_config = tmp_path / "report.toml"
    report_config.write_text(
        '[[review_steps]]\nname="audit"\nkind="visual"\nmode="report"\n', encoding="utf-8"
    )
    invoke(
        "revise",
        root,
        base="r005",
        review=True,
        section="ch-002",
        config=report_config,
    )
    sixth = root / "revisions/r006"
    assert (sixth / "notes.md").read_bytes() == after_image
    assert cloud.requests[-1]["schema"]["properties"]["frames"] == {"type": "null"}
    assert "图文应用契约" not in cloud.requests[-1]["instructions"]
    manifest = read_json(root / ".work/manifest.json")
    assert list(manifest["versions"]) == [f"r{number:03d}" for number in range(1, 7)]
    for number in (2, 3, 4, 6):
        replay_revision(root, f"r{number:03d}")
    reading = tmp_path / "reading"
    invoke("export", root, base="r006", output=reading)
    portable = tmp_path / "portable"
    shutil.copytree(sixth, portable)
    assert not (portable / ".work").exists()
    moved_source = tmp_path / "moved-source.mp4"
    source.rename(moved_source)
    offline = tmp_path / "offline"
    invoke("export", portable, output=offline)
    assert len(cloud.requests) == call_count + 1
    assert (
        read_json(root / ".work/manifest.json") == manifest and digest(moved_source) == source_hash
    )
    for directory in (reading, offline):
        assert (directory / "notes.md").read_bytes() == after_image
        for reference in image_dependencies(after_image):
            assert (directory / reference).is_file()
        html = (directory / "notes.html").read_text(encoding="utf-8")
        assert "<strong>术语（term）</strong>" in html and "<math" in html
        assert "公式后正文使用条件" in html and "<script>" not in html
        assert "引用公式后的说明" in html and "onerror=" not in html
        assert "&lt;script&gt;literal()&lt;/script&gt;" in html
        assert "按**（说明）**操作" in html
        assert "用户保留说明" in html and "用户附录" in html
        text = "\n".join(page.extract_text() for page in PdfReader(directory / "notes.pdf").pages)
        assert "公式后正文使用条件" in text and "用户保留说明" in text and "用户附录" in text
        assert "引用公式后的说明" in text
    assert (reading / "notes.html").read_bytes() == (offline / "notes.html").read_bytes()
    invoke("export", root, output=tmp_path / "wrong-format", format="docx", code=2)
    assert not (tmp_path / "wrong-format").exists()
    logs = (root / ".work/logs/events.jsonl").read_text(encoding="utf-8")
    responses = "".join(
        path.read_text(encoding="utf-8") for path in (root / ".work/model-responses").glob("*.json")
    )
    assert all(
        marker not in logs + responses
        for marker in ("private-thinking-marker", "deepseek-test-secret", "qwen-test-secret")
    )
    assert "deepseek_context" in manifest["config"]
    moved_source.rename(source)
    manual_path = sixth / "notes.md"
    manual = after_image.replace(b"![", b"![changed", 1)
    manual_path.write_bytes(manual)
    first_picture = current_book.chapters[0].blocks[1].id
    result = invoke("revise", root, base="r006", block=first_picture, frame="20", code=2)
    assert "图片行已有手改" in result.output
    assert manual_path.read_bytes() == manual and not (root / "revisions/r007").exists()


@pytest.mark.parametrize(
    "cloud",
    [
        (supplier, failure)
        for supplier in ("deepseek", "qwen")
        for failure in (
            "incomplete",
            "budget",
            "unknown_evidence",
            "injected_path",
            "cancel",
            "review_budget",
            "review_cancel",
        )
    ],
    indirect=True,
)
def test_multichapter_failure_never_registers_success(lesson, tmp_path, cloud):
    """中间章截断、预算耗尽、伪造证据或路径注入均只留诊断，阻止版本与后续修订伪成功。"""
    root = tmp_path / "partial"
    config = configuration(
        tmp_path / "config.toml",
        max_calls={"budget": 2, "review_budget": 6}.get(cloud.failure, 30),
    )
    source_hash = digest(lesson)
    result = invoke(
        "convert",
        lesson,
        output=root,
        config=config,
        provider=cloud.supplier,
        code=130 if cloud.failure in ("cancel", "review_cancel") else 1,
    )
    manifest = read_json(root / ".work/manifest.json")
    if cloud.failure.startswith("review_"):
        assert manifest["status"] == "completed" and set(manifest["versions"]) == {"r001"}
        replay_conversion(root)
    else:
        assert not manifest["versions"]
        assert not any(row["packet"]["operation"] == "review" for row in cloud.requests)
        if cloud.failure == "cancel":
            assert manifest["status"] == "cancelled" and not (root / "notes.md").exists()
            assert (root / "transcript.jsonl").is_file()
        else:
            assert "图文整理部分完成" in result.output and manifest["status"] == "partial"
            assert any(chapter["status"] == "failed" for chapter in manifest["chapters"])
        invoke("revise", root, section="ch-001", instruction="重写", code=2)
    assert not (root / "revisions").exists() and digest(lesson) == source_hash
    assert (
        read_json(root / "usage.json")["input_tokens"]
        == len(cloud.requests) * 1000 + len(cloud.audio) * 400
    )
    assert not list(tmp_path.glob(".partial.*.partial"))
    calls = len(cloud.requests)
    for value in ("-1", "00:60:00", "NaN", "1.1234567"):
        invoke("convert", lesson, output=tmp_path / "invalid", start=value, code=2)
    invoke("convert", lesson, output=lesson.parent, code=2)
    assert not (tmp_path / "invalid").exists() and len(cloud.requests) == calls
