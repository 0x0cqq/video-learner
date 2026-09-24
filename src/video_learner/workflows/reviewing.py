"""对显式版本独立复审，保留初稿并提交可重放的局部修订。"""

import hashlib
import uuid
from collections.abc import Callable
from pathlib import Path

from video_learner.common.config import Config, load_config
from video_learner.common.core import InputError, contained
from video_learner.common.storage import (
    Events,
    atomic_bytes,
    canonical_hash,
    digest,
    directory_lock,
    read_json,
    write_json,
)
from video_learner.common.usage import summarize_usage
from video_learner.media.io import inspect_source
from video_learner.notes.composition import validate_notebook
from video_learner.notes.rendering import (
    AnchorConflict,
    expected_spans,
    export_book,
    image_dependencies,
    local_image,
)
from video_learner.notes.reviewing import apply_review_pass, review_input
from video_learner.providers.base import PROMPT_VERSION, Provider, create_provider
from video_learner.workflows.conversion import extraction_hash, fingerprint_source
from video_learner.workflows.revision import baseline_config, conflict_report, load_baseline


def review(
    workdir: Path,
    *,
    base: str = "r001",
    section: str | None = None,
    config_path: Path | None = None,
    secret: Path | None = None,
    model_provider: str | None = None,
    model: str | None = None,
    provider: Provider | None = None,
    progress: Callable[[dict], None] | None = None,
    usage_report: Callable[[dict], None] | None = None,
) -> Path:
    """复用原始证据复审整份讲义或指定章节；持有与转换、修订相同的目录锁。"""
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
            destination = review_version(root, base, config, events, provider, section)
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
    provider: Provider | None = None,
    section: str | None = None,
) -> Path:
    """在调用方持锁时核验基线、冻结独立复审输入，提交新版本；失败保留原版本。"""
    manifest, baseline, book, snapshot = load_baseline(root, base)
    if extraction_hash(config) != extraction_hash(baseline_config(manifest)):
        raise InputError("提取配置与基线不一致，请重新转换到独立目录")
    current_path = contained(baseline, "notes.md")
    if current_path.stat().st_size > 30_000_000:
        raise InputError("当前 Markdown 超过 30 MB，不能安全进行局部修订")
    current = current_path.read_bytes()
    current_hash = hashlib.sha256(current).hexdigest()
    try:
        spans = expected_spans(current, book)
        original_spans = expected_spans(snapshot, book)
    except AnchorConflict as exc:
        report = conflict_report(root, str(exc))
        raise InputError(f"{exc}；诊断与建议片段：{report}") from exc
    chapters = [c for c in book.chapters if section is None or c.id == section]
    if not chapters:
        raise InputError("指定章节不存在")
    for reference in image_dependencies(current):
        if not contained(baseline, local_image(reference)).is_file():
            raise InputError("基线文档中的图片依赖缺失")
    local_source = read_json(contained(root, ".work/source-local.json"))
    if canonical_hash(local_source.get("files")) != manifest.get("source_hash"):
        raise InputError("本地来源清单与原始指纹不一致")
    source_path = Path(local_source["path"])
    subtitle = Path(local_source["subtitle"]) if local_source.get("subtitle") else None
    source = inspect_source(source_path)
    fingerprints = fingerprint_source(source_path, source, subtitle)
    if fingerprints != local_source["files"] or source.model_dump(
        exclude={"title"}
    ) != book.source.model_dump(exclude={"title"}):
        raise InputError("源素材与基线不一致，请重新转换到独立目录")
    validate_notebook(book, root)
    if current != snapshot:
        book.sync_status = "manual_unverified"
    for chapter in book.chapters:
        for block in chapter.blocks:
            now, old = spans[block.id], original_spans[block.id]
            if current[now.start : now.end] != snapshot[old.start : old.end]:
                block.sync_status = "manual_unverified"
    number = max(int(identity[1:]) for identity in manifest["versions"]) + 1
    revision_id = f"r{number:03d}"
    destination = contained(root, f"revisions/{revision_id}")
    if destination.exists():
        raise InputError("新版本目录已存在但未登记，请保留并人工核对")
    record_id = f"{events.run_id}-{revision_id}"
    record = contained(root, f".work/reviews/{record_id}")
    staging = contained(root, f".work/revision-{uuid.uuid4().hex}.partial")
    staging.mkdir(parents=True, exist_ok=False)
    frozen_book = book.model_copy(deep=True)
    frozen_current = current
    atomic_bytes(record / "baseline.md", current)
    write_json(record / "baseline.json", book.model_dump())
    write_json(
        record / "config.json", config.model_dump(exclude={"secret_file", "asr_secret_file"})
    )
    changes = [f"# {revision_id} 复审修改\n\n基线：{base}\n"]
    active_provider = provider or create_provider(config, events)
    try:
        with events.stage("review"):
            for index, chapter in enumerate(chapters):
                events.emit(
                    "review", "progress", total=len(chapters), completed=index, detail=chapter.id
                )
                frozen, images = review_input(
                    frozen_book, chapter, config, root, frozen_current, revision_id
                )
                write_json(record / f"{chapter.id}.input.json", frozen.model_dump())
                result = active_provider.review(frozen.packet, images)
                write_json(record / f"{chapter.id}.result.json", result.model_dump())
                current = apply_review_pass(book, frozen.packet, result, current)
                changes.extend(
                    f"- {chapter.id} / {item.target_id} / {item.action}：{item.reason}\n"
                    for item in result.findings
                )
                changes.extend(
                    f"- {chapter.id} / {item.frame_id} / {item.decision}"
                    f"（关联 {item.related_block_id or '无'}）：{item.reason}\n"
                    for item in result.frames
                )
            events.emit("review", "progress", total=len(chapters), completed=len(chapters))
            validate_notebook(book, root)
            export_book(book, root, staging, current, base=baseline)
            atomic_bytes(staging / "changes.md", "\n".join(changes).encode("utf-8"))
            if digest(current_path) != current_hash:
                report = conflict_report(root, "复审期间基线被再次编辑", current)
                raise InputError(f"检测到并发编辑；建议文稿：{report}")
            if fingerprint_source(source_path, source, subtitle) != fingerprints:
                raise InputError("复审期间源素材发生变化，未发布版本")
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging.rename(destination)
            atomic_bytes(contained(root, f".work/versions/{revision_id}.generated.md"), current)
            manifest["versions"][revision_id] = {
                "folder": f"revisions/{revision_id}",
                "base": base,
                "status": "completed",
                "markdown_hash": hashlib.sha256(current).hexdigest(),
                "notes_hash": digest(destination / "notes.json"),
                "base_current_hash": current_hash,
                "review_record": record_id,
                "review_sections": [c.id for c in chapters],
                "prompt_version": PROMPT_VERSION,
            }
            write_json(contained(root, ".work/manifest.json"), manifest)
        return destination
    except BaseException as exc:
        if staging.exists():
            write_json(
                staging / "failure.json", {"status": "failed", "error_type": type(exc).__name__}
            )
        raise
