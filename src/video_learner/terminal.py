"""将结构化事件显示为阶段进度；终端表现与业务日志相互独立。"""

import time
from dataclasses import dataclass, field

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.text import Text

from video_learner.common.core import timestamp

STAGES = {
    "prepare": "素材准备",
    "transcribe": "语音识别",
    "sample": "截图提取",
    "scan_frames": "扫描画面",
    "save_frames": "保存截图",
    "plan_chapters": "章节划分",
    "compose": "图文整理",
    "validate_export": "校验与导出",
    "revise": "修订",
}


@dataclass
class RequestMetrics:
    """按真实调用编号统计请求，响应、校验和结束事件不重复计时。"""

    calls: set[int] = field(default_factory=set)
    ended: set[int] = field(default_factory=set)
    durations: dict[int, float] = field(default_factory=dict)
    retries: int = 0

    def observe(self, record: dict) -> None:
        """仅已发出的重试计入请求数；均耗时包括返回响应和失败请求的已知耗时。"""
        call = record.get("call")
        if call is None:
            return
        status = record["status"]
        if status == "running" and call not in self.calls:
            self.calls.add(call)
            self.retries += int(record.get("attempt", 1) > 1)
        elif status in ("received", "completed", "failed", "incomplete"):
            self.ended.add(call)
            if "seconds" in record:
                self.durations.setdefault(call, record["seconds"])

    def describe(self, planned: int | None = None) -> str:
        """总量包含已发生重试，未来切片变化和重试未知，所以始终明确标注预计。"""
        text = f"请求已发 {len(self.calls)} · 已结束 {len(self.ended)}"
        if planned is not None:
            total = max(planned + self.retries, len(self.calls))
            text = f"预计总请求 {total} · " + text + f" · 待发约 {total - len(self.calls)}"
        average = (
            f"{sum(self.durations.values()) / len(self.durations):.1f} 秒"
            if self.durations
            else "—"
        )
        if self.durations and len(self.durations) < len(self.ended):
            average += f"（{len(self.durations)} 次已计时）"
        return text + f" · 重试 {self.retries} · 均耗时 {average}"


class StageProgress(Progress):
    def get_renderables(self):
        """计数、请求统计和当前活动分行显示，窄终端自动折行。"""
        if not self.tasks:
            return
        yield self.make_tasks_table(self.tasks)
        task = self.tasks[0]
        detail = task.fields.get("detail", "")
        activity = task.fields.get("activity", "")
        metrics = task.fields.get("metrics", "")
        if metrics:
            yield Text(metrics)
        elapsed = time.monotonic() - task.fields["run_started"]
        yield Text(f"{detail} {activity} · 总用时 {elapsed:.0f} 秒".strip())


class TerminalProgress:
    def __init__(self, console: Console, verbose: bool = False):
        """交互终端原地刷新；详细模式与重定向输出仅打印普通文本。"""
        self.console = console
        self.verbose = verbose
        self.interactive = console.is_terminal and not verbose
        self.progress = StageProgress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("{task.fields[count]}"),
            console=console,
            disable=not self.interactive,
            transient=True,
            refresh_per_second=4,
        )
        self.task = self.progress.add_task(
            "准备启动", total=None, count="", detail="", run_started=time.monotonic()
        )
        self.stage = ""
        self.started = time.monotonic()
        self.requests = {"transcribe": RequestMetrics(), "compose": RequestMetrics()}
        self.planned: dict[str, int] = {}
        self.pause_cuts = 0
        self.audio_chunks = 0
        self.counts: dict[str, int] = {}
        self.uses_subtitles = False

    def __enter__(self):
        self.started = time.monotonic()
        self.progress.update(self.task, run_started=self.started)
        self.progress.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        """始终清除动态显示，失败或中断时不留下成功进度。"""
        self.progress.stop()
        for stage, metrics in self.requests.items():
            if metrics.calls:
                self.console.print(f"{STAGES[stage]}：{metrics.describe()}")
        outcome = (
            "完成" if exc_type is None else "已中断" if exc_type is KeyboardInterrupt else "未完成"
        )
        self.console.print(f"{outcome} · 总用时 {time.monotonic() - self.started:.1f} 秒")

    def __call__(self, record: dict) -> None:
        """仅消费阶段、进度和请求活动；原始细节仍由 Events 写入日志文件。"""
        raw_stage, status = record["stage"], record["status"]
        stage = raw_stage.split(":", 1)[0]
        if self.verbose:
            seconds = f" ({record['seconds']:.1f}s)" if "seconds" in record else ""
            self.console.print(f"{raw_stage}：{status}{seconds}", markup=False)
        if stage == "conversion_plan":
            self.uses_subtitles = record["uses_subtitles"]
            audio = (
                "使用外部字幕"
                if record["uses_subtitles"]
                else f"音频每片最多 {record['asr_window_seconds']} 秒，默认寻找末尾停顿切分"
            )
            if self.verbose:
                self.console.print(f"音频策略：{audio}。")
                self.console.print(f"章节目标 {record['chapter_seconds']} 秒，转写后对齐边界。")
        elif stage == "chapter_plan":
            self.console.print(f"已划分 {record['chapters']} 章。")
        elif stage == "audio_boundary" and status == "selected":
            self.audio_chunks += 1
            self.pause_cuts += int(record.get("reason") == "pause")
        if status == "progress" and stage in ("scan_frames", "save_frames"):
            self.counts[stage] = record["completed"]
        if stage in STAGES and status in ("running", "progress"):
            # 逐章的内部阶段不重置整个图文进度条。
            if stage != self.stage:
                self.stage = stage
                self.progress.reset(
                    self.task,
                    total=record.get("total"),
                    description=STAGES[stage],
                    count="",
                    detail="",
                    activity="",
                    run_started=self.started,
                    metrics="",
                )
                if not self.verbose and not self.interactive:
                    self.console.print(f"{STAGES[stage]}…")
            if status == "progress":
                completed, total = record["completed"], record["total"]
                if record.get("unit") == "audio":
                    count = f"{completed / total:.0%} · {timestamp(completed)} / {timestamp(total)}"
                else:
                    count = f"{completed}/{total}"
                if record.get("failed"):
                    count += f" · 失败 {record['failed']} 章"
                if stage == "compose":
                    self.planned[stage] = total
                elif "request_total" in record:
                    self.planned[stage] = record["request_total"]
                self.progress.update(
                    self.task,
                    completed=completed,
                    total=total,
                    count=count,
                    detail=record.get("detail", ""),
                    activity="",
                )
        if status in ("completed", "partial", "failed", "cancelled") and stage in STAGES:
            if ":" not in raw_stage and status in ("completed", "partial"):
                if not self.verbose:
                    self.console.print(self.stage_summary(stage, record))
                elif stage == "transcribe":
                    self.console.print(f"已采用停顿切点 {self.pause_cuts} 处；静音保留。")
            elif status == "failed" and stage == "compose":
                self.console.print(
                    f"章节 {raw_stage.split(':')[-1]} 未完成。",
                    style="yellow",
                    markup=False,
                )
        if stage in ("model_call", "asr_model_call"):
            if status == "running":
                self.progress.update(self.task, activity="等待模型响应")
            elif status == "retrying":
                self.progress.update(self.task, activity="等待重试")
            elif status in ("completed", "failed", "incomplete"):
                self.progress.update(self.task, activity="处理响应")
        elif stage == "model_stream":
            activity = {"thinking": "模型思考中", "answering": "生成正文中"}.get(status)
            if activity:
                self.progress.update(self.task, activity=activity)
        if stage in (
            "asr_model_call",
            "asr_model_usage",
            "model_call",
            "model_usage",
            "model_validation",
        ):
            group = "transcribe" if stage.startswith("asr_") else "compose"
            self.requests[group].observe(record)
        group = "compose" if self.stage == "revise" else self.stage
        if group in self.requests:
            metrics = self.requests[group].describe(self.planned.get(group))
            self.progress.update(self.task, metrics=metrics)

    def stage_summary(self, stage: str, record: dict) -> str:
        """用已完成阶段的实际计数生成摘要；部分失败不显示成功标记。"""
        partial = record["status"] == "partial"
        text = f"{'△' if partial else '✓'} {STAGES[stage]}{'部分完成' if partial else '完成'}"
        if stage == "transcribe":
            text += " · 外部字幕" if self.uses_subtitles else f" · 音频 {self.audio_chunks} 片"
        elif stage == "sample":
            text += (
                f" · 扫描 {self.counts.get('scan_frames', 0)} 帧"
                f" · 候选 {self.counts.get('save_frames', 0)} 张"
            )
        elif stage == "compose":
            text += f" · 成功 {record['chapters']} 章 · 插图 {record['figures']} 张"
            if partial:
                text += f" · 失败 {record['failed']} 章"
        if stage in self.requests and self.requests[stage].calls:
            metrics = self.requests[stage]
            text += f" · 请求 {len(metrics.calls)} 次（重试 {metrics.retries} 次）"
        if "seconds" in record:
            text += f" · 耗时 {record['seconds']:.1f} 秒"
        return text
