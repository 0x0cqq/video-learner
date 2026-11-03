"""确定性协议响应夹具，供模型故障定位和全流程测试共用。"""

import io
import sys
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace as NS
from unittest.mock import patch

from video_learner.cli import main
from video_learner.common.config import Config
from video_learner.common.schemas import Draft, DraftBlock, FrameAssessment, ReviewResult
from video_learner.common.storage import Events
from video_learner.notes.generation import NoteGenerator
from video_learner.providers.deepseek import DeepSeekProvider
from video_learner.providers.qwen import QwenProvider


def packet():
    return {
        "start_us": 0,
        "end_us": 1_000_000,
        "transcript": [{"id": "tr-1", "start_us": 0, "end_us": 1_000_000, "text": "字幕"}],
        "frames": [],
        "allow_ai_additions": False,
    }


VALID = (
    '{"title":"章节","blocks":[{"kind":"text","body":"忠实整理",'
    '"category":"original","evidence_ids":["tr-1"],"frame_id":null}],"review":[]}'
)


def response(text=VALID, status="completed"):
    return NS(status=status, output_text=text, usage=NS(input_tokens=100, output_tokens=50))


class DeepSeekClient:
    def __init__(self, responses):
        self.responses = self
        self.values = iter(responses)
        self.requests = []

    def create(self, **kwargs):
        """保留实际请求，按预定顺序返回响应或抛出服务错误。"""
        self.requests.append(kwargs)
        value = next(self.values)
        if isinstance(value, Exception):
            raise value
        return value


def chunk(content=None, reasoning=None, finish=None, usage=None):
    """构造正文、思考或仅用量的流式块；usage 尾块刻意不带 choices，以覆盖真实接口形态。"""
    return NS(
        choices=[NS(delta=NS(content=content, reasoning_content=reasoning), finish_reason=finish)]
        if usage is None
        else [],
        usage=usage,
    )


class Stream:
    def __init__(self, chunks):
        self.chunks, self.closed = chunks, False

    def __iter__(self):
        """按顺序交付流式块，遇异常实例就在迭代中抛出，用于模拟收到部分正文后断线。"""
        for value in self.chunks:
            if isinstance(value, Exception):
                raise value
            yield value

    def close(self):
        self.closed = True


class QwenClient:
    def __init__(self, streams):
        self.chat = self.completions = self
        self.streams, self.requests = iter(streams), []

    def create(self, **kwargs):
        """保留实际请求，模拟建立流时的响应或服务错误。"""
        self.requests.append(kwargs)
        value = next(self.streams)
        if isinstance(value, Exception):
            raise value
        return value


def draft_block(body, evidence, *, frame=None, category="original"):
    """统一有证据的测试块，图片以真实候选 ID 确定类型和关联。"""
    return DraftBlock(
        kind="figure" if frame else "text",
        body=body,
        category=category,
        evidence_ids=[evidence],
        frame_id=frame,
    )


class DeterministicGenerator:
    def review(self, packet, images):
        """保留原配图与疑点，按职能返回空修改，隔离模型质量。"""
        step = packet.get("review_step", {})
        if step.get("kind") == "content" or step.get("mode") == "report":
            return ReviewResult(frames=None, findings=[])
        selected = {b["frame_id"] for b in packet["chapter"]["blocks"] if b["kind"] == "figure"}
        return ReviewResult(
            frames=[
                FrameAssessment(
                    frame_id=frame["id"],
                    decision="use" if frame["id"] in selected else "omit",
                    related_block_id=packet["chapter"]["blocks"][0]["id"],
                    transcript_ids=[],
                    reason="自造画面测试；保持初稿配图。",
                )
                for frame in packet["frames"]
            ],
            findings=[],
        )

    def compose(self, packet, images):
        """固定引用首条转写和首张候选图，隔离云端质量波动，仅验证转换的证据与文件契约。"""
        evidence = packet["transcript"][0]["id"]
        frame = packet["frames"][0]["id"]
        return Draft(
            title="自造媒体验证",
            blocks=[
                draft_block("自造声音与变化画面。", evidence),
                draft_block("变化后的原视频画面。", frame, frame=frame),
            ],
            review=["这是离线测试替身，不是课程质量评测。"],
        )


def session(tmp_path, supplier, replies, **settings):
    """用两家的真实适配器和确定性响应核对同一业务策略，不发网络请求。"""
    config, events = Config(provider=supplier, **settings), Events(tmp_path)
    if supplier == "deepseek":
        client = DeepSeekClient(
            [text if isinstance(text, Exception) else response(text) for text in replies]
        )
        adapter = DeepSeekProvider(config, events, client)
    else:
        client = QwenClient(
            [
                text
                if isinstance(text, (Exception, Stream))
                else Stream([chunk(content=text, finish="stop")])
                for text in replies
            ]
        )
        adapter = QwenProvider(config, events, client)
    return NoteGenerator(
        config, events, adapter, response_directory=tmp_path / ".work/model-responses"
    ), client


def invoke(*arguments, code=0, **options):
    """经 CLI 执行业务命令，失败时保留完整终端输出帮助定位阶段。"""
    flags = []
    for name, value in options.items():
        flags.extend(["--" + name.replace("_", "-")])
        if value is not True:
            flags.append(str(value))
    stdout, stderr, exit_code = io.StringIO(), io.StringIO(), 0
    with (
        redirect_stdout(stdout),
        redirect_stderr(stderr),
        patch.object(sys, "argv", ["video-learner", *(str(value) for value in arguments), *flags]),
    ):
        try:
            main()
        except SystemExit as error:
            exit_code = error.code
    result = NS(stdout=stdout.getvalue(), stderr=stderr.getvalue())
    result.output = result.stdout + result.stderr
    assert exit_code == code, result.output
    return result
