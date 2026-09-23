import io
import json

import pytest
from rich.console import Console

from video_learner.common.storage import Events
from video_learner.terminal import RequestMetrics, TerminalProgress


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
    assert "本次请求" not in rendered


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_failure_and_cancellation_do_not_report_success(error):
    """异常退出先关闭动态显示，保留正确终态并继续传播原异常。"""
    output = io.StringIO()
    with pytest.raises(error), TerminalProgress(Console(file=output)):
        raise error()
    assert output.getvalue().startswith("已中断" if error is KeyboardInterrupt else "未完成")


def test_verbose_and_retry_output():
    """详细模式保留原始事件，默认模式不单独刷出重试次数。"""
    output = io.StringIO()
    with TerminalProgress(Console(file=output), verbose=True) as display:
        display({"stage": "model_usage", "status": "received", "seconds": 1})
        display({"stage": "model_call", "status": "retrying", "attempt": 1, "max_retries": 2})
    assert "model_usage：received" in output.getvalue()
    assert "model_call：retrying" in output.getvalue()
    output = io.StringIO()
    with TerminalProgress(Console(file=output)) as display:
        display({"stage": "model_call", "status": "retrying", "attempt": 1, "max_retries": 2})
    assert "重试" not in output.getvalue()


def test_request_metrics_count_attempts_and_time_each_call_once():
    """失败、返回与校验可能重复报告同一调用；均耗时只计一次，未发出的重试不计数。"""
    metrics = RequestMetrics()
    metrics.observe({"status": "running", "call": 1, "attempt": 1})
    metrics.observe({"status": "failed", "call": 1, "seconds": 6})
    metrics.observe({"status": "retrying", "attempt": 1})
    assert metrics.retries == 0
    metrics.observe({"status": "running", "call": 2, "attempt": 2})
    metrics.observe({"status": "received", "call": 2, "seconds": 2})
    metrics.observe({"status": "failed", "call": 2, "seconds": 2.5})
    metrics.observe({"status": "running", "call": 3, "attempt": 1})
    text = metrics.describe(10)
    assert "预计总请求 11" in text and "待发约 8" in text
    assert "请求已发 3" in text and "已结束 2" in text
    assert "重试 1" in text and "均耗时 4.0 秒" in text


@pytest.mark.parametrize("verbose", [False, True])
def test_strategy_and_chapter_summary_are_visible(verbose):
    """策略细节只在详细模式显示，默认仍报告实际章数。"""
    output = io.StringIO()
    display = TerminalProgress(Console(file=output), verbose=verbose)
    display(
        {
            "stage": "conversion_plan",
            "status": "ready",
            "chapter_seconds": 180,
            "asr_window_seconds": 30,
            "uses_subtitles": False,
        }
    )
    display({"stage": "chapter_plan", "status": "ready", "chapters": 34})
    display(
        {
            "stage": "transcribe",
            "status": "progress",
            "completed": 0,
            "total": 100,
            "unit": "audio",
            "request_total": 4,
        }
    )
    display({"stage": "audio_boundary", "status": "selected", "reason": "pause"})
    display({"stage": "transcribe", "status": "completed", "seconds": 1})
    text = output.getvalue()
    assert ("默认寻找末尾停顿切分" in text) == verbose
    assert ("章节目标 180 秒" in text) == verbose
    assert "已划分 34 章" in text
    assert ("已采用停顿切点 1 处" in text) == verbose


def test_stage_summaries_use_actual_counts_and_preserve_partial_status():
    """片数含无文字切片、请求含重试；部分图文结果不能显示全成功。"""
    output = io.StringIO()
    display = TerminalProgress(Console(file=output, width=180))
    for reason in ("pause", "limit"):
        display({"stage": "audio_boundary", "status": "selected", "reason": reason})
    for call, attempt in ((1, 1), (2, 2), (3, 1)):
        display({"stage": "asr_model_call", "status": "running", "call": call, "attempt": attempt})
    display({"stage": "transcribe", "status": "completed", "seconds": 4})
    for stage, count in (("scan_frames", 100), ("save_frames", 12)):
        display({"stage": stage, "status": "progress", "completed": count, "total": count})
    display({"stage": "sample", "status": "completed", "seconds": 5})
    display(
        {
            "stage": "compose",
            "status": "partial",
            "chapters": 2,
            "failed": 1,
            "figures": 6,
            "seconds": 20,
        }
    )
    text = output.getvalue()
    assert "音频 2 片 · 请求 3 次（重试 1 次） · 耗时 4.0 秒" in text
    assert "扫描 100 帧 · 候选 12 张 · 耗时 5.0 秒" in text
    assert "图文整理部分完成 · 成功 2 章 · 插图 6 张 · 失败 1 章" in text
    assert "✓ 图文整理完成" not in text
