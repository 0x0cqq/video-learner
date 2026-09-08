"""从真实转换日志定位耗时；可离线回放本地阶段，不调用模型或改写已有讲义。"""

import argparse
import cProfile
import io
import json
import math
import pstats
import statistics
import time
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from video_learner.common.core import US, InputError, contained
from video_learner.common.schemas import Draft, Notebook
from video_learner.common.storage import atomic_bytes, digest, read_json, write_json
from video_learner.media.evidence import audio_window, sample_frames
from video_learner.notes.composition import validate_notebook
from video_learner.notes.rendering import export_book, render_notes
from video_learner.providers.asr import encode_wav
from video_learner.workflows.revision import baseline_config


def distribution(values: list[float]) -> dict:
    """概括已记录请求的分布；P95 使用最近秩，空样本明确返回空对象。"""
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "count": len(values),
        "sum_seconds": sum(values),
        "p50_seconds": statistics.median(values),
        "p95_seconds": ordered[math.ceil(len(values) * 0.95) - 1],
        "max_seconds": ordered[-1],
    }


def summarize_events(records: list[dict]) -> dict:
    """汇总首次转换，排除后续修订；嵌套事件不重复计入阶段总时长。"""
    stages, calls, failed, asr, audio = {}, {}, set(), [], []
    failures = Counter()
    for record in records:
        stage, status = record["stage"], record["status"]
        if stage.startswith("revise:"):
            break
        if status == "completed" and (
            stage in ("prepare", "transcribe", "sample", "plan_chapters", "validate_export")
            or stage.startswith("compose:")
        ):
            stages[stage] = record["seconds"]
        if stage == "model_usage":
            calls[record["call"]] = record
        if stage in ("model_call", "model_validation") and status in ("failed", "incomplete"):
            failed.add(record["call"])
            failures[record.get("error_type", stage + ":" + status)] += 1
            if "seconds" in record:
                calls.setdefault(record["call"], record)
        if stage == "asr_model_usage":
            asr.append(record["seconds"])
        if stage.startswith("audio_decode:") and status == "completed":
            audio.append(record["seconds"])
    requests = [record["seconds"] for record in calls.values()]
    failed_seconds = sum(calls[identity]["seconds"] for identity in failed if identity in calls)
    groups = {
        "prepare": stages.get("prepare", 0),
        "transcribe": stages.get("transcribe", 0),
        "sample": stages.get("sample", 0),
        "plan_chapters": stages.get("plan_chapters", 0),
        "compose": sum(value for key, value in stages.items() if key.startswith("compose:")),
        "validate_export": stages.get("validate_export", 0),
    }
    return {
        "stage_seconds": groups,
        "stage_seconds_total": sum(groups.values()),
        "model_requests": distribution(requests),
        "failed_model_requests": len(failed),
        "failed_model_seconds_measured": failed_seconds,
        "failed_model_requests_without_duration": len(failed - calls.keys()),
        "failure_types": dict(failures),
        "asr_requests": distribution(asr),
        "asr_requests_over_10s": len([value for value in asr if value > 10]),
        "asr_seconds_over_10s": sum(value for value in asr if value > 10),
        "audio_prepare_seconds": sum(audio),
        "unmeasured_wall_time": "阶段间清单写入、预检和最终提交未包含在旧日志阶段总和内",
    }


def summarize_run(workdir: Path) -> dict:
    """检查完整转换的日志及原始响应，只记录错误类别和数量，不复制课程正文。"""
    manifest = read_json(contained(workdir, ".work/manifest.json"))
    if manifest.get("status") != "completed":
        raise InputError("此分析入口要求已完成的转换工作目录")
    records = [
        json.loads(line)
        for line in contained(workdir, ".work/logs/events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    result = summarize_events(records)
    shapes = Counter()
    for path in contained(workdir, ".work/model-responses").glob("*.json"):
        raw = path.read_text(encoding="utf-8")
        try:
            Draft.model_validate_json(raw)
            shapes["valid_structure"] += 1
        except ValidationError:
            shapes["empty_array" if raw.strip() == "[]" else "other_invalid"] += 1
    result["raw_response_shapes_all_versions"] = dict(shapes)
    result["video_seconds"] = (manifest["range"][1] - manifest["range"][0]) / US
    result["model"] = manifest["config"]["model"]
    return result


def profile_call(name: str, action, destination: Path) -> dict:
    """测量墙钟和进程 CPU 时间，并保存可用 pstats 打开的逐函数性能数据。"""
    profiler = cProfile.Profile()
    started, cpu_started = time.perf_counter(), time.process_time()
    profiler.runcall(action)
    elapsed, cpu = time.perf_counter() - started, time.process_time() - cpu_started
    profiler.dump_stats(str(contained(destination, f"{name}.prof")))
    buffer = io.StringIO()
    pstats.Stats(profiler, stream=buffer).strip_dirs().sort_stats("cumulative").print_stats(25)
    atomic_bytes(contained(destination, f"{name}.txt"), buffer.getvalue().encode("utf-8"))
    print(f"{destination.name}/{name}: wall={elapsed:.3f}s cpu={cpu:.3f}s", flush=True)
    return {"wall_seconds": elapsed, "process_cpu_seconds": cpu}


def replay_local(workdir: Path, destination: Path, sample_seconds: int) -> dict:
    """复用已生成证据回放本地阶段，写入新的 profiling 目录，验证原讲义字节未改变。"""
    manifest = read_json(contained(workdir, ".work/manifest.json"))
    config = baseline_config(manifest)
    book = Notebook.model_validate(read_json(contained(workdir, "notes.json")))
    source = Path(read_json(contained(workdir, ".work/source-local.json"))["path"]).resolve()
    source_root = source if source.is_dir() else source.parent
    if destination.is_relative_to(source_root) or source_root.is_relative_to(destination):
        raise InputError("profiling 目录不能与源素材目录重叠")
    before = {name: digest(contained(workdir, name)) for name in ("notes.md", "notes.json")}
    begin = book.start_us + (book.end_us - book.start_us) // 2
    stop = min(book.end_us, begin + sample_seconds * US)
    measured = {"sample_range_us": [begin, stop], "cold_cache_controlled": False}

    def prepare_audio():
        """按生产用的音频窗口解码并编码 WAV，剔除云端识别等待。"""
        for start in range(begin, stop, config.asr_window_seconds * US):
            end = min(stop, start + config.asr_window_seconds * US)
            encode_wav(audio_window(source, book.source, start, end))

    measured["audio_prepare"] = profile_call("audio_prepare", prepare_audio, destination)
    measured["sample_frames"] = profile_call(
        "sample_frames",
        lambda: sample_frames(source, book.source, begin, stop, config, destination / "sample"),
        destination,
    )
    measured["validate_notebook"] = profile_call(
        "validate_notebook", lambda: validate_notebook(book, workdir), destination
    )
    measured["render_notes"] = profile_call("render_notes", lambda: render_notes(book), destination)
    measured["export_book"] = profile_call(
        "export_book", lambda: export_book(book, workdir, destination / "export"), destination
    )
    assert before == {name: digest(contained(workdir, name)) for name in before}
    measured["original_notes_unchanged"] = True
    return measured


def render_report(reports: dict) -> str:
    """生成可读的阶段和请求统计，明确历史计时、本地回放与未知项的边界。"""
    lines = [
        "# 转换性能分析",
        "",
        "阶段总和来自原始日志；嵌套请求不重复计时。它不包含旧日志未记录的阶段间开销。",
        "云端请求时间含上传、服务等待和返回数据，不能全部归因于模型思考。",
        "",
        "| Case | 阶段总和 | 转写 | 采样 | 图文整理 | 校验导出 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, report in reports.items():
        stages = report["stage_seconds"]
        values = [report["stage_seconds_total"]] + [
            stages[key] for key in ("transcribe", "sample", "compose", "validate_export")
        ]
        lines.append(f"| {name} | " + " | ".join(f"{value:.1f}s" for value in values) + " |")
    for name, report in reports.items():
        lines += ["", f"## {name}", ""]
        total = report["stage_seconds_total"]
        model, asr = report["model_requests"], report["asr_requests"]
        remote = model.get("sum_seconds", 0) + asr.get("sum_seconds", 0)
        lines.append(f"云端请求合计 {remote:.1f}s，占已计时阶段 {remote / total:.1%}。")
        lines.append(
            f"图文失败请求 {report['failed_model_requests']} 次，已记录耗时 "
            f"{report['failed_model_seconds_measured']:.1f}s；"
            f"另有 {report['failed_model_requests_without_duration']} 次失败缺少请求耗时，未估算。"
        )
        if asr:
            lines.append(
                f"ASR P50 / P95 / 最大请求耗时为 {asr['p50_seconds']:.2f}s / "
                f"{asr['p95_seconds']:.2f}s / {asr['max_seconds']:.2f}s；"
                f"超过 10 秒的 {report['asr_requests_over_10s']} 次请求累计 "
                f"{report['asr_seconds_over_10s']:.1f}s。"
            )
        if "local_replay" in report:
            lines += [
                "",
                "本地回放（未调用模型，未控制冷缓存，cProfile 自身有开销）：",
                "",
                "| 操作 | 墙钟时间 | 进程 CPU 时间 |",
                "| --- | ---: | ---: |",
            ]
            for operation, values in report["local_replay"].items():
                if isinstance(values, dict):
                    lines.append(
                        f"| {operation} | {values['wall_seconds']:.3f}s | "
                        f"{values['process_cpu_seconds']:.3f}s |"
                    )
    lines += [
        "",
        "原始 JSON、逐函数 .prof 和按累计时间排序的 .txt 位于本报告目录。",
        "本地音频/采样仅回放所记录的片段，校验/渲染/导出回放整份讲义；不能直接当成新的整课基准。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """创建新的分析目录，可选执行离线 cProfile，不接受覆盖原转换目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdirs", nargs="+", type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--local", action="store_true", help="离线回放媒体与导出阶段")
    parser.add_argument("--sample-seconds", type=int, default=120, choices=(30, 60, 120, 300))
    args = parser.parse_args()
    destination = args.report_dir.resolve()
    roots = [path.resolve() for path in args.workdirs]
    if len({path.name for path in roots}) != len(roots):
        parser.error("工作目录名称须唯一，以区分报告")
    if any(destination.is_relative_to(path) or path.is_relative_to(destination) for path in roots):
        parser.error("报告目录不能与原转换目录重叠")
    for root in roots:
        source = Path(read_json(contained(root, ".work/source-local.json"))["path"]).resolve()
        source_root = source if source.is_dir() else source.parent
        if destination.is_relative_to(source_root) or source_root.is_relative_to(destination):
            parser.error("报告目录不能与源素材目录重叠")
    destination.mkdir(parents=True, exist_ok=False)
    reports = {}
    for root in roots:
        result = summarize_run(root)
        reports[root.name] = result
        if args.local:
            folder = contained(destination, root.name)
            folder.mkdir()
            result["local_replay"] = replay_local(root, folder, args.sample_seconds)
        write_json(contained(destination, "summary.json"), reports)
    atomic_bytes(contained(destination, "report.md"), render_report(reports).encode("utf-8"))
    print(destination / "report.md")


if __name__ == "__main__":
    main()
