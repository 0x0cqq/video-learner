"""对显式版本独立复审，保留初稿并提交可重放的局部修订。"""

import uuid
from collections.abc import Callable
from pathlib import Path

from video_learner.common.config import Config, load_config
from video_learner.common.core import InputError, contained
from video_learner.common.storage import (
    Events,
    atomic_bytes,
    directory_lock,
    write_json,
)
from video_learner.common.usage import summarize_usage
from video_learner.exports import export_book
from video_learner.exports.pdf import font_path
from video_learner.notes.composition import validate_notebook
from video_learner.notes.generation import Generator, create_generator
from video_learner.notes.prompts import PROMPT_VERSION
from video_learner.notes.reviewing import ReviewFunction, apply_review_pass, review_input
from video_learner.workflows.versions import (
    baseline_config,
    commit_revision,
    extraction_hash,
    load_baseline,
    load_editable_baseline,
    next_revision,
    verify_source,
)


def review(
    workdir: Path,
    *,
    base: str = "r001",
    section: str | None = None,
    config_path: Path | None = None,
    secret: Path | None = None,
    model_provider: str | None = None,
    model: str | None = None,
    generator: Generator | None = None,
    reviewers: dict[str, ReviewFunction] | None = None,
    progress: Callable[[dict], None] | None = None,
    usage_report: Callable[[dict], None] | None = None,
    pdf_font: Path | None = None,
) -> Path:
    """复用原始证据复审整份讲义或指定章节；持有与转换、修订相同的目录锁。"""
    font_path(pdf_font)
    root = workdir.resolve()
    if not contained(root, ".work/manifest.json").is_file():
        raise InputError("此目录没有本工具的工作清单")
    with directory_lock(root.parent / f".{root.name}.lock"):
        manifest, _, _, _ = load_baseline(root, base)
        config = load_config(
            config_path,
            base=baseline_config(manifest).model_dump(),
            secret_file=str(secret) if secret else None,
            provider=model_provider,
            model=model,
        )
        events = Events(root, progress)
        try:
            destination = review_version(
                root,
                base,
                config,
                events,
                generator,
                section,
                reviewers,
                pdf_font=pdf_font,
            )
        finally:
            report = summarize_usage(events.usage_events, config)
            write_json(contained(root, f".work/review-usage/{events.run_id}.json"), report)
            if usage_report:
                usage_report(report)
        write_json(destination / "usage.json", report)
        return destination


def review_version(
    root: Path,
    base: str,
    config: Config,
    events: Events,
    generator: Generator | None = None,
    section: str | None = None,
    reviewers: dict[str, ReviewFunction] | None = None,
    pdf_font: Path | None = None,
) -> Path:
    """在调用方持锁时核验基线、冻结独立复审输入，提交新版本；失败保留原版本。"""
    editable = load_editable_baseline(root, base)
    manifest, baseline, book = editable.manifest, editable.directory, editable.book
    if extraction_hash(config) != extraction_hash(baseline_config(manifest)):
        raise InputError("提取配置与基线不一致，请重新转换到独立目录")
    current = editable.current
    chapters = [c for c in book.chapters if section is None or c.id == section]
    if not chapters:
        raise InputError("指定章节不存在")
    reviewers = reviewers or {}
    if set(reviewers) - {step.name for step in config.review_steps}:
        raise InputError("自定义复审函数名称须在 review_steps 中配置")
    source_state = verify_source(root, manifest, book)
    validate_notebook(book, root)
    revision_id, destination = next_revision(root, manifest)
    record_id = f"{events.run_id}-{revision_id}"
    record = contained(root, f".work/reviews/{record_id}")
    staging = contained(root, f".work/revision-{uuid.uuid4().hex}.partial")
    staging.mkdir(parents=True, exist_ok=False)
    atomic_bytes(record / "baseline.md", current)
    write_json(record / "baseline.json", book.model_dump())
    write_json(
        record / "config.json", config.model_dump(exclude={"secret_file", "asr_secret_file"})
    )
    changes = [f"# {revision_id} 复审修改\n\n基线：{base}\n"]
    active_generator = generator
    try:
        with events.stage("review"):
            total = len(chapters) * len(config.review_steps)
            completed = 0
            for step in config.review_steps:
                # 同一轮各章使用固定快照；下一轮统一读取本轮的完整结果。
                frozen_book, frozen_current = book.model_copy(deep=True), current
                reviewer = reviewers.get(step.name)
                if reviewer is None:
                    active_generator = active_generator or create_generator(
                        config, events, response_directory=contained(root, ".work/model-responses")
                    )
                    reviewer = active_generator.review
                changes.append(f"\n## {step.name}（{step.kind} / {step.mode}）\n")
                for target in chapters:
                    chapter = next(c for c in frozen_book.chapters if c.id == target.id)
                    events.emit(
                        "review",
                        "progress",
                        total=total,
                        completed=completed,
                        detail=f"{step.name} / {chapter.id}",
                    )
                    frozen, images = review_input(
                        frozen_book, chapter, config, root, frozen_current, revision_id, step
                    )
                    write_json(record / step.name / f"{chapter.id}.input.json", frozen.model_dump())
                    # 自定义函数也只拿独立输入；应用和重放始终使用未改写的冻结包。
                    result = reviewer(frozen.model_copy(deep=True).packet, images)
                    book, current = apply_review_pass(book, frozen.packet, result, current)
                    write_json(
                        record / step.name / f"{chapter.id}.result.json", result.model_dump()
                    )
                    changes.extend(
                        f"- {chapter.id} / {item.target_id} / {item.action}：{item.reason}\n"
                        for item in result.findings
                    )
                    changes.extend(
                        f"- {chapter.id} / {item.frame_id} / {item.decision}"
                        f"（关联 {item.related_block_id or '无'}）：{item.reason}\n"
                        for item in result.frames or []
                    )
                    changes.extend(
                        f"- {chapter.id} / 已解决疑点 {identity}\n"
                        for identity in result.resolved_review_ids
                    )
                    completed += 1
            events.emit("review", "progress", total=total, completed=completed)
            validate_notebook(book, root)
            export_book(book, root, staging, current, base=baseline, pdf_font=pdf_font)
            atomic_bytes(staging / "changes.md", "\n".join(changes).encode("utf-8"))
            commit_revision(
                root,
                editable,
                source_state,
                staging,
                destination,
                current,
                {
                    "review_record": record_id,
                    "review_sections": [c.id for c in chapters],
                    "review_steps": [step.name for step in config.review_steps],
                    "prompt_version": PROMPT_VERSION,
                },
                current,
            )
        return destination
    except BaseException as exc:
        if staging.exists():
            write_json(
                staging / "failure.json", {"status": "failed", "error_type": type(exc).__name__}
            )
        raise
