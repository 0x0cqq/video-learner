import io
import json

from rich.console import Console

from video_learner.common.storage import Events
from video_learner.terminal import TerminalProgress


def test_redirected_output_is_short_but_log_keeps_details(tmp_path):
    """一百次请求只打印阶段摘要，结构化用量和所有细节仍保存在日志中。"""
    output = io.StringIO()
    with TerminalProgress(Console(file=output, force_terminal=False)) as display:
        events = Events(tmp_path, display)
        with events.stage("transcribe"):
            for call in range(100):
                events.emit("asr_model_call", "running", call=call)
                events.emit("asr_model_usage", "received", call=call, input_tokens=10)
                events.emit("transcribe", "progress", completed=call + 1, total=100, unit="audio")
    lines = output.getvalue().splitlines()
    assert len(lines) <= 6
    assert "语音识别" in lines[0] and "完成" in lines[-1]
    assert "\x1b" not in output.getvalue()
    records = [json.loads(line) for line in events.path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 302 and len(events.usage_events) == 200


def test_interactive_progress_keeps_failures_and_metrics_visible():
    """章节失败不计入成功进度，窄终端展示请求计划、统计与模型活动。"""
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, width=80, color_system=None)
    display = TerminalProgress(console)
    display(
        {
            "stage": "compose",
            "status": "progress",
            "completed": 2,
            "total": 5,
            "failed": 1,
            "detail": "第 4 章",
        }
    )
    display({"stage": "compose:ch-004", "status": "running"})
    display({"stage": "model_call", "status": "running", "call": 1})
    display({"stage": "model_stream", "status": "thinking"})
    console.print(display.progress.get_renderable())
    rendered = output.getvalue()
    assert "2/5" in rendered and "失败 1 章" in rendered
    assert "第 4 章" in rendered and "模型思考中" in rendered
    assert "预计总请求 5" in rendered and "请求已发 1" in rendered and "待发约 4" in rendered
