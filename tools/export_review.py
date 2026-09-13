"""把人工校订的结构化讲义导出为独立阅读副本；复用本地证据，零 API。"""

import argparse
import uuid
from pathlib import Path

from video_learner.common.config import Config
from video_learner.common.core import InputError, output_path
from video_learner.common.schemas import Draft, Notebook
from video_learner.common.storage import read_json
from video_learner.notes.composition import evidence_packet, validate_draft, validate_notebook
from video_learner.notes.rendering import export_book, render_notes


def export_review(root: Path, edited: Path, output: Path) -> Path:
    """拒绝证据变更及覆盖手改；校验全部章节后发布副本，不登记转换或修订版本。"""
    root = root.resolve()
    target = output_path(root, output)
    baseline = Notebook.model_validate(read_json(root / "notes.json"))
    book = Notebook.model_validate(read_json(edited))
    if (root / "notes.md").read_bytes() != render_notes(baseline):
        raise InputError("基线 Markdown 有手改，请先显式合并到校订稿，避免遗漏手改")
    for key in ("source", "transcript", "frames", "start_us", "end_us"):
        if getattr(book, key) != getattr(baseline, key):
            raise InputError(f"审阅副本不得改变原始证据或范围：{key}")
    if [(c.id, c.start_us, c.end_us) for c in book.chapters] != [
        (c.id, c.start_us, c.end_us) for c in baseline.chapters
    ]:
        raise InputError("审阅副本必须保留原章节范围和 ID")
    config = Config.model_validate(read_json(root / ".work/manifest.json")["config"])
    for chapter in book.chapters:
        if chapter.status != "completed":
            raise InputError(f"审阅副本仍有未完成章节：{chapter.id}")
        packet, _ = evidence_packet(book, chapter, config, root)
        # 人工审阅可看本章所有已存原帧，不受单次模型请求的图片数量限制。
        packet["frames"] = [
            {"id": f.id, "at_us": f.at_us}
            for f in book.frames
            if chapter.start_us <= f.at_us < chapter.end_us
        ]
        draft = Draft(
            title=chapter.title,
            blocks=[
                b.model_dump(exclude={"id", "chapter_id", "sync_status"}) for b in chapter.blocks
            ],
            review=[],
        )
        validate_draft(draft, packet)
    validate_notebook(book, root)
    staging = target.with_name(f".{target.name}.review-{uuid.uuid4().hex}")
    staging.mkdir(parents=True, exist_ok=False)
    export_book(book, root, staging)
    staging.rename(target)
    return target


def main() -> None:
    """显式指定原工作目录、人工校订 JSON 和全新输出目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("edited", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(export_review(args.workdir, args.edited, args.output))


if __name__ == "__main__":
    main()
