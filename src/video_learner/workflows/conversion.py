"""P0 用例编排；CLI 仅转换参数并展示阶段事件。"""

import hashlib
import shutil
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from video_learner.common.config import Config
from video_learner.common.core import InputError, TaskError, contained, output_path, time_range
from video_learner.common.schemas import Notebook, ReviewItem, Source
from video_learner.common.storage import (
    Events,
    atomic_bytes,
    canonical_hash,
    digest,
    directory_lock,
    write_json,
)
from video_learner.common.usage import summarize_usage
from video_learner.media.evidence import load_subtitles, sample_frames, save_transcript
from video_learner.media.io import inspect_source, source_file, track_of
from video_learner.notes.composition import (
    apply_chapter_draft,
    composition_input,
    plan_chapters,
    validate_notebook,
)
from video_learner.notes.rendering import export_book
from video_learner.providers.asr import transcribe, validate_asr_config
from video_learner.providers.base import (
    PROMPT_VERSION,
    Provider,
    create_provider,
    validate_provider_config,
)

EXTRACTION_FIELDS = {
    "asr_language",
    "asr_qwen_model",
    "asr_window_seconds",
    "sample_seconds",
    "chapter_seconds",
    "max_images_per_chapter",
    "image_change_threshold",
}


def extraction_hash(config: Config) -> str:
    """只对影响证据提取的配置求指纹，允许修订时更换图文模型而复用原始证据。"""
    fields = config.model_dump(include=EXTRACTION_FIELDS)
    if config.asr_backend == "local":
        fields.update(
            config.model_dump(
                include={
                    "asr_backend",
                    "asr_device",
                    "asr_local_model",
                    "asr_beam_size",
                }
            )
        )
    return canonical_hash(fields)


def fingerprint_source(path: Path, source: Source, subtitle: Path | None = None) -> dict:
    """为实际使用的媒体、显式字幕及课程元数据生成绝对路径到内容哈希的映射。"""
    files = {source_file(path, track).resolve() for track in source.tracks}
    if subtitle:
        files.add(subtitle.resolve())
    if path.is_dir() and (path / "videoInfo.json").is_file():
        files.add((path / "videoInfo.json").resolve())
    return {str(file): digest(file) for file in sorted(files)}


def convert(
    source_path: Path,
    output: Path,
    config: Config,
    start_us: int = 0,
    end_us: int | None = None,
    subtitle: Path | None = None,
    progress: Callable[[dict], None] | None = None,
    provider: Provider | None = None,
    usage_report: Callable[[dict], None] | None = None,
    force: bool = False,
) -> Path:
    """编排单视频转换，在独占锁内提取证据、组织章节并提交新输出目录。

    输入与凭据预检通过后才创建工作目录；仅全部章节完成才登记 r001。
    force 在预检通过并持锁后删除原输出；后续转换失败不会恢复被删除的旧结果。
    失败保留诊断产物并抛异常，这些文件不代表可恢复任务或可修订的成功版本。
    """
    source_path = source_path.resolve()
    target = output_path(source_path, output, force=force)
    source = inspect_source(source_path)
    if source.diagnostics:
        raise InputError("素材检查发现问题：" + "；".join(source.diagnostics))
    track_of(source, "audio")
    start_us, end_us = time_range(start_us, end_us, source.duration_us)
    subtitle_segments = None
    subtitle_report = None
    if subtitle:
        subtitle = subtitle.resolve()
        subtitle_segments, subtitle_report = load_subtitles(
            subtitle,
            start_us,
            end_us,
            source.duration_us,
        )
    if provider is None:
        validate_provider_config(config)
    if subtitle_segments is None:
        validate_asr_config(config, start_us, end_us)
    if force:
        for dependency in (
            subtitle,
            config.secret_file,
            config.asr_secret_file,
            config.asr_local_model,
        ):
            if dependency and Path(dependency).resolve().is_relative_to(target):
                raise InputError("字幕、凭据或本地模型位于输出目录内，不能强制删除")
    # 锁放在输出目录旁，既不提前创建输出，也能与后续 revise 使用同一把锁。
    lock_path = target.parent / f".{target.name}.lock"
    with directory_lock(lock_path):
        # 删除前重新解析并校验同一个绝对目标，防止等待锁期间路径被替换。
        if output_path(source_path, output, force=force) != target:
            raise InputError("输出路径在预检后发生变化，停止转换")
        if force and target.exists():
            shutil.rmtree(target)
        staging = target.parent / f".{target.name}.{uuid.uuid4().hex}.partial"
        staging.mkdir(parents=True, exist_ok=False)
        events = Events(staging, progress)
        events.emit(
            "conversion_plan",
            "ready",
            chapter_seconds=config.chapter_seconds,
            asr_window_seconds=config.asr_window_seconds,
            uses_subtitles=subtitle_segments is not None,
        )
        manifest = {
            "schema_version": 2,
            "status": "running",
            "source_hash": None,
            "extraction_hash": extraction_hash(config),
            "extraction_version": 4,
            "prompt_version": PROMPT_VERSION,
            "config": config.model_dump(exclude={"secret_file", "asr_secret_file"}),
            "range": [start_us, end_us],
            "versions": {},
            "chapters": [],
        }
        book = Notebook(
            title=source.title,
            start_us=start_us,
            end_us=end_us,
            source=source,
            chapters=[],
            transcript=[],
            frames=[],
        )
        try:
            with events.stage("prepare"):
                fingerprints = fingerprint_source(source_path, source, subtitle)
                manifest["source_hash"] = canonical_hash(fingerprints)
                write_json(
                    contained(staging, ".work/source-local.json"),
                    {
                        "path": str(source_path),
                        "files": fingerprints,
                        "subtitle": str(subtitle) if subtitle else None,
                    },
                )
                write_json(contained(staging, ".work/manifest.json"), manifest)
            with events.stage("transcribe"):
                book.transcript = (
                    subtitle_segments
                    if subtitle_segments is not None
                    else transcribe(
                        source_path,
                        source,
                        start_us,
                        end_us,
                        config,
                        staging,
                        events,
                    )
                )
                save_transcript(staging, book.transcript)
                if any(s.alignment == "audio_window" for s in book.transcript):
                    book.review.append(
                        ReviewItem(
                            reason="语音来源按实际音频切片区间引用，"
                            "不是逐句对齐。切片边界可能截断词句，请结合原图和音频核对。",
                            start_us=start_us,
                            end_us=end_us,
                        )
                    )
                if subtitle_report:
                    write_json(contained(staging, ".work/subtitle-check.json"), subtitle_report)
                    book.review.append(
                        ReviewItem(
                            reason=f"外部字幕覆盖率 {subtitle_report['coverage_ratio']:.1%}；"
                            "首尾和中间同步尚需人工核对",
                            start_us=start_us,
                            end_us=end_us,
                        )
                    )
                if not book.transcript:
                    book.review.append(
                        ReviewItem(
                            reason="所选片段未识别到语音，以下内容仅根据画面整理，请核对音频是否正常",
                            start_us=start_us,
                            end_us=end_us,
                        )
                    )
            with events.stage("sample"):
                book.frames = sample_frames(
                    source_path, source, start_us, end_us, config, staging, events
                )
            with events.stage("plan_chapters"):
                book.chapters = plan_chapters(start_us, end_us, config, book.transcript)
                events.emit("chapter_plan", "ready", chapters=len(book.chapters))
                manifest["chapters"] = [c.model_dump() for c in book.chapters]
                write_json(contained(staging, ".work/manifest.json"), manifest)
                write_json(contained(staging, ".work/evidence.json"), book.model_dump())
            active_provider = provider or create_provider(config, events)
            compose_started = time.monotonic()
            for index, chapter in enumerate(book.chapters, 1):
                events.emit(
                    "compose",
                    "progress",
                    total=len(book.chapters),
                    completed=sum(c.status == "completed" for c in book.chapters),
                    failed=sum(c.status == "failed" for c in book.chapters),
                    detail=f"第 {index} 章",
                )
                try:
                    with events.stage(f"compose:{chapter.id}"):
                        frozen, images = composition_input(book, chapter, config, staging)
                        write_json(
                            contained(staging, f".work/composition-inputs/{chapter.id}.json"),
                            frozen.model_dump(),
                        )
                        draft = active_provider.compose(frozen.packet, images)
                        apply_chapter_draft(book, chapter, frozen.packet, draft)
                        write_json(
                            contained(staging, f".work/composition-drafts/{chapter.id}.json"),
                            draft.model_dump(),
                        )
                except TaskError as exc:
                    # 单章失败仍继续处理其他章节，但清除该章块，避免半成品被误读为完成。
                    chapter.status = "failed"
                    chapter.blocks = []
                    book.review.append(
                        ReviewItem(
                            reason=f"本章未完成：{exc}",
                            start_us=chapter.start_us,
                            end_us=chapter.end_us,
                        )
                    )
                manifest["chapters"] = [c.model_dump() for c in book.chapters]
                write_json(contained(staging, ".work/manifest.json"), manifest)
                write_json(contained(staging, ".work/composed.json"), book.model_dump())
            events.emit(
                "compose",
                "progress",
                total=len(book.chapters),
                completed=sum(c.status == "completed" for c in book.chapters),
                failed=sum(c.status == "failed" for c in book.chapters),
            )
            events.emit(
                "compose",
                "completed" if all(c.status == "completed" for c in book.chapters) else "partial",
                seconds=time.monotonic() - compose_started,
                chapters=sum(c.status == "completed" for c in book.chapters),
                failed=sum(c.status == "failed" for c in book.chapters),
                figures=sum(b.kind == "figure" for c in book.chapters for b in c.blocks),
            )
            with events.stage("validate_export"):
                validate_notebook(book, staging)
                current = export_book(book, staging, staging)
                atomic_bytes(contained(staging, ".work/versions/r001.generated.md"), current)
                manifest["status"] = (
                    "completed"
                    if all(chapter.status == "completed" for chapter in book.chapters)
                    else "partial"
                )
                if manifest["status"] == "completed":
                    manifest["versions"]["r001"] = {
                        "folder": ".",
                        "base": None,
                        "status": "completed",
                        "markdown_hash": hashlib.sha256(current).hexdigest(),
                        "notes_hash": digest(contained(staging, "notes.json")),
                    }
                write_json(contained(staging, ".work/manifest.json"), manifest)
            # 模型请求可能较长，提交前重算素材指纹，防止发布期间输入已被并发改写。
            if fingerprint_source(source_path, source, subtitle) != fingerprints:
                raise InputError("处理期间源素材发生变化，请重新转换到独立目录")
            staging.rename(target)
            if manifest["status"] != "completed":
                raise TaskError(f"部分章节未完成，诊断和部分产物保存在 {target}")
            return target
        except BaseException as exc:
            if staging.exists():
                manifest["status"] = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
                manifest["error_type"] = type(exc).__name__
                manifest["versions"] = {}
                write_json(contained(staging, ".work/manifest.json"), manifest)
                # 失败产物只供诊断，不能作为可恢复任务或成功修订基线。
                if not target.exists():
                    staging.rename(target)
            raise
        finally:
            # 提交或失败移动目录后仍写入本次运行的用量，不混入以后修订的请求。
            report = summarize_usage(events.usage_events, config)
            folder = staging if staging.exists() else target
            write_json(contained(folder, "usage.json"), report)
            if usage_report:
                usage_report(report)
