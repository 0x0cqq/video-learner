"""复用已保存的转写与图片，优先读取原始章节包；--live 才调用图文模型。"""

import argparse
from pathlib import Path

from video_learner.common.config import Config
from video_learner.common.core import InputError, TaskError, contained, output_path
from video_learner.common.schemas import CompositionInput, Notebook
from video_learner.common.storage import Events, digest, read_json, write_json
from video_learner.common.usage import summarize_usage
from video_learner.notes.composition import evidence_packet
from video_learner.providers.base import create_provider


def main() -> None:
    """复用章节证据比较图文策略，不重跑识别、不抽帧、不覆盖成功讲义。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sections", nargs="+", required=True)
    parser.add_argument("--context", choices=["history", "chapter"], default="history")
    parser.add_argument("--secret", type=Path)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    root = args.workdir.resolve()
    target = output_path(root, args.output)
    book = Notebook.model_validate(read_json(root / "notes.json"))
    config = Config.model_validate(read_json(root / ".work/manifest.json")["config"])
    selected = [c for c in book.chapters if c.id in args.sections]
    if len(selected) != len(args.sections):
        raise InputError("章节不存在或重复，请从 sources.md 选择明确 ID")
    if args.live and len(selected) > 3:
        raise InputError("单次真实对照最多选择三章，整课检查请先复用已有产物")
    config = config.model_copy(
        update={
            "deepseek_context": args.context,
            "max_calls": len(selected),
            "max_retries": 0,
            "secret_file": str(args.secret.resolve()) if args.secret else None,
        }
    )
    target.mkdir(parents=True, exist_ok=False)
    events = Events(target)
    service = create_provider(config, events) if args.live else None
    try:
        for chapter in selected:
            recorded = contained(root, f".work/composition-inputs/{chapter.id}.json")
            if recorded.is_file():
                frozen = CompositionInput.model_validate(read_json(recorded))
                packet = frozen.packet
                images = []
                for image in frozen.images:
                    path = contained(root, image.path)
                    if digest(path) != image.sha256:
                        raise TaskError(f"{chapter.id} 的图片证据已经变化")
                    images.append((image.id, path))
            else:
                packet, images = evidence_packet(book, chapter, config, root)
                print(f"{chapter.id}: 原任务未保存输入，按现有索引重组证据包", flush=True)
            write_json(target / f"{chapter.id}.packet.json", packet)
            print(
                f"{chapter.id}: {len(packet['transcript'])} 段转写 / {len(images)} 图"
                f" / {'真实请求' if service else '仅离线准备'}",
                flush=True,
            )
            if service:
                draft = service.compose(packet, images)
                write_json(target / f"{chapter.id}.draft.json", draft.model_dump())
    finally:
        report = summarize_usage(events.usage_events, config)
        write_json(target / "usage.json", report)
        print(f"估算费用：{report['estimated_known_cost']}", flush=True)


if __name__ == "__main__":
    main()
