"""只读统计整课候选、实际提供给模型的图片与最终配图，辅助定位漏图阶段。"""

import argparse
from itertools import pairwise
from pathlib import Path

from video_learner.common.core import US, InputError, contained, output_path, timestamp
from video_learner.common.schemas import CompositionInput, Notebook
from video_learner.common.storage import atomic_bytes, read_json, write_json


def coverage_report(root: Path, revision: str = "r001") -> dict:
    """按章节比较三层图片覆盖；缺少冻结输入时明确留空，不推测历史请求。"""
    book = Notebook.model_validate(read_json(root / "notes.json"))
    record_id = None
    if revision != "r001":
        from video_learner.workflows.revision import load_baseline

        manifest, _, book, _ = load_baseline(root, revision)
        record_id = manifest["versions"][revision].get("review_record")
    frames = {frame.id: frame for frame in book.frames}
    chapters = []
    selected_times = set()
    for chapter in book.chapters:
        candidates = [
            frame.id for frame in book.frames if chapter.start_us <= frame.at_us < chapter.end_us
        ]
        frozen_path = contained(root, f".work/composition-inputs/{chapter.id}.json")
        if revision != "r001":
            frozen_path = (
                contained(root, f".work/reviews/{record_id}/{chapter.id}.input.json")
                if record_id
                else None
            )
        offered = None
        if frozen_path is not None and frozen_path.is_file():
            frozen = CompositionInput.model_validate(read_json(frozen_path))
            offered = [image.id for image in frozen.images]
        selected = [block.frame_id for block in chapter.blocks if block.kind == "figure"]
        times = [frames[identity].at_us for identity in selected]
        selected_times.update(times)
        chapters.append(
            {
                "id": chapter.id,
                "title": chapter.title,
                "start_us": chapter.start_us,
                "end_us": chapter.end_us,
                "candidate_ids": candidates,
                "offered_ids": offered,
                "selected_ids": selected,
                "selected_times_us": times,
                "text_characters": sum(
                    len(block.body) for block in chapter.blocks if block.kind == "text"
                ),
            }
        )
    boundaries = sorted({book.start_us, book.end_us, *selected_times})
    gaps = sorted(
        (
            {"start_us": start, "end_us": end, "seconds": (end - start) / US}
            for start, end in pairwise(boundaries)
        ),
        key=lambda item: item["seconds"],
        reverse=True,
    )
    return {
        "revision": revision,
        "range_us": [book.start_us, book.end_us],
        "candidate_count": len(book.frames),
        "figure_count": sum(len(chapter["selected_ids"]) for chapter in chapters),
        "chapters": chapters,
        "largest_figure_gaps": gaps[:5],
        "limits": "统计来自 notes.json，未反向解析 Markdown 手改。空白区间包括课程首尾，"
        "仅用于定位审阅；图片数量和时间间隔不是语义质量分数。"
        "未保存冻结输入的旧产物，其实际送图数量记为未知。",
    }


def render_report(report: dict) -> str:
    """生成可定位到原视频的覆盖表，保留旧产物缺少输入记录的限制。"""
    lines = [
        "# 图片覆盖审阅",
        "",
        report["limits"],
        "",
        f"候选 {report['candidate_count']} 张，正文配图 {report['figure_count']} 张。",
        "",
        "| 章节 | 原课区间 | 候选 | 实际送图 | 正文配图 | 正文字符 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for chapter in report["chapters"]:
        offered = chapter["offered_ids"]
        title = chapter["title"].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {chapter['id']} {title} | {timestamp(chapter['start_us'])}–"
            f"{timestamp(chapter['end_us'])} | {len(chapter['candidate_ids'])} | "
            f"{len(offered) if offered is not None else '未知'} | "
            f"{len(chapter['selected_ids'])} | {chapter['text_characters']} |"
        )
    lines += ["", "## 最长配图时间间隔", ""]
    for gap in report["largest_figure_gaps"]:
        lines.append(
            f"- {timestamp(gap['start_us'])}–{timestamp(gap['end_us'])}："
            f"{gap['seconds'] / 60:.2f} 分钟"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """读取已落盘的讲义并写入独立的新报告目录，保护素材及原转换结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--revision", default="r001", help="显式版本，复审版读取其实际复审送图记录")
    args = parser.parse_args()
    root = args.workdir.resolve()
    target = output_path(root, args.output)
    local_source = root / ".work/source-local.json"
    if local_source.is_file():
        source = Path(read_json(local_source)["path"]).resolve()
        source_root = source if source.is_dir() else source.parent
        if target.is_relative_to(source_root) or source_root.is_relative_to(target):
            raise InputError("报告目录不能与源素材目录重叠")
    report = coverage_report(root, args.revision)
    target.mkdir(parents=True, exist_ok=False)
    write_json(target / "coverage.json", report)
    atomic_bytes(target / "coverage.md", render_report(report).encode("utf-8"))
    print(target / "coverage.md")


if __name__ == "__main__":
    main()
