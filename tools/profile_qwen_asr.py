"""在独立 Transformers 环境测量开源 Qwen3-ASR，复用已保存的真实 WAV 窗口。"""

import argparse
import json
import re
import threading
import time
from importlib.metadata import version
from pathlib import Path


def write_json(path: Path, value: object) -> None:
    """完整序列化后原子提交评估文件，避免中断留下伪完整报告。"""
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    """只读短样本并测量两轮推理；保留原课区间，下载与模型加载分开计时。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="含 transcript.jsonl 和 .work/audio 的目录")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", choices=["0.6B", "1.7B"], default="0.6B")
    parser.add_argument("--model-path", type=Path, help="已从官方仓库下载的原生 HF 权重目录")
    parser.add_argument("--batch-size", type=int, choices=[1, 2, 3, 6], default=1)
    parser.add_argument("--language", default="Chinese")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        parser.error("输入与输出目录必须互不包含")
    rows = [
        json.loads(line)
        for line in (source / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or any(row["end_us"] <= row["start_us"] for row in rows):
        parser.error("输入须包含有效的真实音频窗口")
    duration = sum(row["end_us"] - row["start_us"] for row in rows) / 1_000_000
    if not 5 <= duration <= 180:
        parser.error("测量音频总长须为 5–180 秒")
    paths = []
    for row in rows:
        if not re.fullmatch(r"audio-\d+", row["chunk_id"]):
            parser.error("无效的音频窗口 ID")
        path = (source / ".work/audio" / (row["chunk_id"] + ".wav")).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            parser.error("音频窗口缺失或越出输入目录")
        paths.append(path)
    output.mkdir(parents=True, exist_ok=False)

    import psutil
    import pynvml
    import soundfile as sf
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("此次测量需要可用的 CUDA PyTorch")
    torch.set_num_threads(4)
    model_id = f"Qwen/Qwen3-ASR-{args.model}-hf"
    print(f"准备官方权重：{model_id}", flush=True)
    started = time.perf_counter()
    snapshot = (
        str(args.model_path.resolve())
        if args.model_path
        else snapshot_download(
            model_id, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"]
        )
    )
    download_seconds = time.perf_counter() - started
    print(f"权重就绪，用时 {download_seconds:.2f} 秒；开始加载 GPU 模型", flush=True)

    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
    process = psutil.Process()
    stop = threading.Event()
    samples = []

    def sample() -> dict:
        """记录进程内存和整卡显存，后者包括其他应用。"""
        return {
            "rss_bytes": process.memory_info().rss,
            "gpu_board_bytes": pynvml.nvmlDeviceGetMemoryInfo(gpu).used,
        }

    def monitor() -> None:
        """固定间隔采样资源，进度始终由主推理流程直接打印。"""
        while not stop.wait(0.1):
            samples.append(sample())

    baseline = sample()
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    runs = []
    try:
        started = time.perf_counter()
        processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
        model = (
            AutoModelForMultimodalLM.from_pretrained(
                snapshot, dtype=torch.bfloat16, local_files_only=True
            )
            .to("cuda")
            .eval()
        )
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - started
        print(f"加载完成：{load_seconds:.2f} 秒；开始两轮转写", flush=True)
        for repeat in (1, 2):
            result = []
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started, cpu = time.perf_counter(), time.process_time()
            for offset in range(0, len(rows), args.batch_size):
                batch = rows[offset : offset + args.batch_size]
                audio = []
                for path in paths[offset : offset + args.batch_size]:
                    waveform, rate = sf.read(path, dtype="float32")
                    if rate != 16000 or waveform.ndim != 1:
                        raise ValueError("共用音频窗口须为 16kHz 单声道")
                    audio.append(waveform)
                inputs = processor.apply_transcription_request(
                    audio=audio, language=[args.language] * len(batch)
                ).to(model.device, model.dtype)
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=512, do_sample=False)
                generated = generated[:, inputs["input_ids"].shape[1] :]
                if generated.shape[1] >= 512:
                    raise RuntimeError("输出达到 token 上限，需核对截断或重复生成")
                texts = processor.decode(generated, return_format="transcription_only")
                if len(texts) != len(batch):
                    raise RuntimeError("返回数量与输入窗口不一致")
                for row, text in zip(batch, texts, strict=True):
                    result.append({**row, "text": text})
                print(f"第 {repeat} 轮：完成 {len(result)}/{len(rows)} 窗口", flush=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            row = {
                "run": repeat,
                "wall_seconds": elapsed,
                "cpu_seconds": time.process_time() - cpu,
                "real_time_factor": elapsed / duration,
                "gpu_allocated_peak_bytes": torch.cuda.max_memory_allocated(),
                "gpu_reserved_peak_bytes": torch.cuda.max_memory_reserved(),
            }
            write_json(output / f"transcript-{repeat}.json", result)
            runs.append(row)
            print(json.dumps(row), flush=True)
    finally:
        samples.append(sample())
        stop.set()
        thread.join()
        pynvml.nvmlShutdown()
    report = {
        "model": model_id,
        "revision": None if args.model_path else Path(snapshot).name,
        "weights_path": snapshot,
        "versions": {name: version(name) for name in ("torch", "transformers", "soundfile")},
        "device": torch.cuda.get_device_name(),
        "dtype": "bfloat16",
        "batch_size": args.batch_size,
        "language": args.language,
        "audio_seconds": duration,
        "source": str(source),
        "download_seconds": download_seconds,
        "model_load_seconds": load_seconds,
        "baseline": baseline,
        "peak_rss_bytes": max(s["rss_bytes"] for s in samples),
        "peak_gpu_board_bytes": max(s["gpu_board_bytes"] for s in samples),
        "timing_scope": "已存 WAV 读取、预处理、GPU 推理；下载、加载、结果落盘另计",
        "memory_scope": "PyTorch allocated/reserved；100ms 采样 RSS/整卡占用含其他进程",
        "runs": runs,
    }
    write_json(output / "profile.json", report)
    print(f"测量完成：{output / 'profile.json'}", flush=True)


if __name__ == "__main__":
    main()
