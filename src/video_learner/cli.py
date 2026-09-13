"""参数和终端显示；业务处理不依赖 Typer。"""

import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from video_learner.common.core import InputError, TaskError, timestamp
from video_learner.media.io import inspect_source
from video_learner.terminal import TerminalProgress

app = typer.Typer(
    no_args_is_help=False,
    invoke_without_command=True,
    help="将单视频整理为可核对、可修订的图文 Markdown。",
)
console = Console(stderr=True)


@app.callback()
def root(ctx: typer.Context) -> None:
    """无子命令时显示帮助，作为正常调用结束。"""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command("inspect")
def inspect_command(
    source: Annotated[Path, typer.Argument(help="本地视频或单课时缓存目录")],
    as_json: bool = typer.Option(False, "--json", help="机器可读 JSON"),
    decode: bool = typer.Option(False, help="抽查前、中、后部音视频解码"),
    full: bool = typer.Option(False, help="顺序解码全部音视频帧"),
) -> None:
    """检查本地视频或单课时缓存，输出媒体信息与诊断。"""
    # 只转换显示格式：JSON 走 stdout，人工提示走 stderr；有诊断仍输出结果并返回非零码。
    result = inspect_source(source, decode=decode, full=full)
    if as_json:
        typer.echo(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    else:
        console.print(f"{result.title} | {timestamp(result.duration_us)} | {result.adapter}")
        for track in result.tracks:
            console.print(f"{track.kind}: {track.codec} {track.width or ''}×{track.height or ''}")
        console.print(f"字幕：{', '.join(result.subtitles) or '未发现独立 SRT/VTT'}")
        console.print(
            f"元数据下载完成：{result.metadata_complete}；媒体打开及轨道探测：{result.opened}；"
            f"片段解码：{result.sampled_decode}；全片验证：{result.full_verified}"
        )
        for diagnostic in result.diagnostics:
            console.print(diagnostic, style="yellow")
    if result.diagnostics:
        raise typer.Exit(1)


@app.command("convert")
def convert_command(
    source: Annotated[Path, typer.Argument(help="本地视频或单课时缓存")],
    output: Annotated[Path, typer.Option(help="输出目录；已存在时需用 -f 删除覆盖")],
    force: bool = typer.Option(
        False, "--force", "-f", help="删除已有输出目录后重新转换，含旧版本和手改"
    ),
    start: str = "0",
    end: str | None = None,
    profile: str | None = None,
    instruction: str | None = None,
    subtitle: Path | None = None,
    config: Path | None = None,
    model: str | None = None,
    provider: Annotated[str | None, typer.Option(help="图文模型供应商 deepseek 或 qwen")] = None,
    asr_backend: Annotated[str | None, typer.Option(help="ASR 后端 qwen 或 local")] = None,
    asr_device: Annotated[str | None, typer.Option(help="本地 ASR 设备 cpu 或 cuda")] = None,
    asr_secret: Annotated[Path | None, typer.Option(help="Qwen ASR 的独立凭据文件")] = None,
    jobs: Annotated[
        int | None, typer.Option(min=1, max=32, help="ASR 并发数，默认 1；图文仍串行")
    ] = None,
    secret: Path | None = None,
    allow_ai_additions: Annotated[
        bool | None, typer.Option("--allow-ai-additions/--no-ai-additions")
    ] = None,
    verbose: bool = typer.Option(False, "--verbose", help="显示详细阶段事件"),
) -> None:
    """转换完整视频或 [start,end) 片段，输出图文讲义。"""
    from video_learner.common.config import load_config
    from video_learner.common.core import parse_time
    from video_learner.workflows.conversion import convert

    settings = load_config(
        config,
        profile=profile,
        instruction=instruction,
        model=model,
        provider=provider,
        asr_backend=asr_backend,
        asr_device=asr_device,
        asr_secret_file=str(asr_secret) if asr_secret else None,
        jobs=jobs,
        secret_file=str(secret) if secret else None,
        allow_ai_additions=allow_ai_additions,
    )
    reports = []
    try:
        with TerminalProgress(console, verbose) as progress:
            path = convert(
                source,
                output,
                settings,
                parse_time(start),
                parse_time(end) if end else None,
                subtitle,
                progress=progress,
                usage_report=reports.append,
                force=force,
            )
    finally:
        for report in reports:
            display_usage(report)
    typer.echo(str(path / "notes.md"))


def display_usage(report: dict) -> None:
    """向 stderr 展示本次转换 token 与估价，缺失用量和部分费用不显示为完整总价。"""
    table = Table(title="Token meter · 本次转换")
    for heading in ("模型", "请求", "输入 token", "输出 token", "音频秒数", "Estimated 费用"):
        table.add_column(heading)
    for row in report["models"]:
        cost = row["estimated_known_cost"]
        amount = "未估算" if cost is None else f"{cost:.6f} {row['currency']}"
        if cost is not None and row["unpriced_requests"]:
            amount += "（部分）"
        table.add_row(
            row["model"],
            str(row["requests"]),
            f"{row['input_tokens']:,}",
            f"{row['output_tokens']:,}",
            f"{row['audio_seconds']:g}" if row["audio_seconds"] else "—",
            amount,
        )
    console.print(table)
    total = report["input_tokens"] + report["output_tokens"]
    console.print(f"已记录 token：{total:,}；详细用量和计价配置：usage.json")
    if report["missing_token_usage"]:
        console.print(f"另有 {report['missing_token_usage']} 次请求缺少完整 token 用量。")
    if any(row["unknown_cache_requests"] for row in report["models"]):
        console.print("部分请求未返回缓存细分，相关输入按普通输入价格估算。")
    if report["estimated_known_cost"]:
        label = "Estimated 总费用" if report["estimate_complete"] else "Estimated 已知部分费用"
        amounts = " + ".join(f"{v:.6f} {k}" for k, v in report["estimated_known_cost"].items())
        console.print(f"{label}：{amounts}（按配置单价估算，以供应商账单为准）")
    else:
        console.print("费用未估算：需要模型单价及相应计费用量。")


@app.command("revise")
def revise_command(
    workdir: Annotated[Path, typer.Argument(help="保留 .work 的转换工作目录")],
    base: str = "r001",
    section: str | None = None,
    block: str | None = None,
    instruction: str | None = None,
    frame: str | None = None,
    config: Path | None = None,
    secret: Path | None = None,
    provider: Annotated[str | None, typer.Option(help="图文模型供应商 deepseek 或 qwen")] = None,
    model: str | None = None,
    allow_ai_additions: Annotated[
        bool | None, typer.Option("--allow-ai-additions/--no-ai-additions")
    ] = None,
    verbose: bool = typer.Option(False, "--verbose", help="显示详细阶段事件"),
) -> None:
    """基于指定版本，只修订目标章节/段落或精确换图。"""
    from video_learner.common.core import parse_time
    from video_learner.workflows.revision import revise

    with TerminalProgress(console, verbose) as progress:
        path = revise(
            workdir,
            base=base,
            section=section,
            block=block,
            instruction=instruction,
            at_us=parse_time(frame) if frame else None,
            config_path=config,
            secret=secret,
            model_provider=provider,
            model=model,
            allow_ai_additions=allow_ai_additions,
            progress=progress,
        )
    typer.echo(str(path / "notes.md"))


def main() -> None:
    # 统一进程入口的编码和退出码；Windows 重定向可能继承旧代码页，需显式改为 UTF-8。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        result = app(standalone_mode=False)
        if isinstance(result, int):
            raise SystemExit(result)
    except typer.Exit as exc:
        raise SystemExit(exc.exit_code) from None
    except typer.TyperException as exc:
        console.print(f"参数错误：{exc.format_message()}", style="red", markup=False)
        raise SystemExit(exc.exit_code) from None
    except InputError as exc:
        console.print(f"输入错误：{exc}", style="red", markup=False)
        raise SystemExit(2) from exc
    except (TaskError, OSError) as exc:
        console.print(f"运行失败：{exc}", style="red", markup=False)
        raise SystemExit(1) from exc
    except (KeyboardInterrupt, typer.Abort):
        console.print("用户已中断；已提交的文件保留。P0 不支持自动续跑。")
        raise SystemExit(130) from None
    except Exception as exc:
        console.print(
            f"运行失败 ({type(exc).__name__})；请检查阶段日志。未登记成功的目录不能续跑。",
            style="red",
            markup=False,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
