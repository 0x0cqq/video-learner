"""在指定短片段测量本地 ASR：加载、冷/热运行、CPU、RSS 与整卡显存；无付费请求。"""

import argparse
import json
import threading
import time
from pathlib import Path

import psutil

from video_learner.common.config import Config
from video_learner.common.core import US, output_path, parse_time, time_range
from video_learner.common.storage import Events, write_json
from video_learner.media.evidence import save_transcript
from video_learner.media.io import inspect_source
from video_learner.providers.asr import transcribe
from video_learner.providers.local_asr import LocalASR


def main() -> None:
    """一次测量一种配置，模型只加载一次；重复运行使用独立输出且保留原课时间。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", default="60")
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--model")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--repeat", type=int, choices=[1, 2, 3], default=2)
    args = parser.parse_args()
    if not 5 <= args.seconds <= 180:
        parser.error("profiling 片段须为 5–180 秒")
    source = args.source.resolve()
    output = output_path(source, args.output)
    info = inspect_source(source)
    start_us = parse_time(args.start)
    begin, end = time_range(start_us, start_us + args.seconds * US, info.duration_us)
    config = Config(
        asr_backend="local",
        asr_device=args.device,
        asr_local_model=args.model,
        asr_cpu_threads=args.threads,
        jobs=args.jobs,
        asr_language=args.language,
    )
    output.mkdir(parents=True, exist_ok=False)
    process = psutil.Process()
    measurements = []
    stop = threading.Event()
    gpu = None
    if args.device == "cuda":
        import pynvml

        pynvml.nvmlInit()
        gpu = pynvml.nvmlDeviceGetHandleByIndex(0)

    def sample() -> dict:
        """WDDM 下用整卡占用作为保守观测值，明确区别于本进程显存。"""
        return {
            "rss_bytes": process.memory_info().rss,
            "gpu_board_bytes": pynvml.nvmlDeviceGetMemoryInfo(gpu).used if gpu else None,
        }

    def monitor() -> None:
        """固定 100 ms 采样内存，不扫描日志推测进度。"""
        while not stop.wait(0.1):
            measurements.append(sample())

    baseline = sample()
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    service = None
    runs = []
    started = time.monotonic()
    try:
        print(
            f"加载本地模型：device={args.device}, threads={args.threads}, jobs={args.jobs}",
            flush=True,
        )
        service = LocalASR(config, Events(output))
        load_seconds = time.monotonic() - started
        for index in range(args.repeat):
            folder = output / f"run-{index + 1}"
            events = Events(folder)
            print(f"短片段转写 {index + 1}/{args.repeat}：{args.seconds} 秒", flush=True)
            start, cpu = time.monotonic(), time.process_time()
            with events.stage("transcribe"):
                result = transcribe(source, info, begin, end, config, folder, events, service)
            elapsed, cpu_seconds = time.monotonic() - start, time.process_time() - cpu
            save_transcript(folder, result)
            row = {
                "run": index + 1,
                "wall_seconds": elapsed,
                "cpu_seconds": cpu_seconds,
                "cpu_core_equivalent": cpu_seconds / elapsed,
                "real_time_factor": elapsed / args.seconds,
                "segments": len(result),
            }
            runs.append(row)
            print(json.dumps(row), flush=True)
    finally:
        measurements.append(sample())
        if service is not None:
            service.close()
        stop.set()
        thread.join()
        if gpu:
            pynvml.nvmlShutdown()
    report = {
        "config": config.model_dump(exclude={"prices"}),
        "range_us": [begin, end],
        "model_load_seconds": load_seconds,
        "runs": runs,
        "baseline": baseline,
        "peak_rss_bytes": max(r["rss_bytes"] for r in measurements),
        "peak_gpu_board_bytes": max(r["gpu_board_bytes"] for r in measurements) if gpu else None,
        "memory_scope": "进程 RSS；显存为整卡占用，含桌面与其他程序，100 ms 采样",
    }
    write_json(output / "profile.json", report)
    print(f"测量完成：{output / 'profile.json'}", flush=True)


if __name__ == "__main__":
    main()
