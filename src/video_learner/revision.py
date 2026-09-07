"""显式基线、目标范围替换与手改保护。"""

import hashlib
import re
import uuid
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from .application import EXTRACTION_FIELDS, extraction_hash, fingerprint_source
from .composition import evidence_packet, validate_draft, validate_notebook
from .config import Config, load_config, merge_provider_settings
from .core import InputError, TaskError, contained
from .documents import (
    AnchorConflict,
    copy_dependencies,
    expected_spans,
    export_book,
    frame_of,
    image_dependencies,
    local_image,
    render_block,
    render_chapter,
    stage_assets,
)
from .evidence import register_frame
from .media import inspect_source
from .provider import Provider, create_provider
from .schemas import NoteBlock, Notebook, ReviewItem
from .storage import (
    Events,
    atomic_bytes,
    canonical_hash,
    digest,
    directory_lock,
    read_json,
    write_json,
)


def baseline_config(manifest: dict) -> Config:
    """先按清单记录的指纹版本验证提取配置，再加载修订仍需的设置。

    旧版包含已移除的本地 ASR 字段，须按旧规则校验后再剔除，不能用新默认值重算旧指纹。
    """
    settings = manifest.get("config")
    if not isinstance(settings, dict):
        raise InputError("工作清单配置无效")
    obsolete = {
        "asr_model",
        "asr_compute_type",
        "asr_mode",
        "asr_provider",
        "audio_chunk_seconds",
        "audio_overlap_seconds",
    }
    version = manifest.get("extraction_version", 1)
    if version == 1:
        fields = {k: v for k, v in settings.items() if k in EXTRACTION_FIELDS | obsolete}
        if fields.get("asr_mode", "standard") == "standard":
            fields.pop("asr_mode", None)
        if fields.get("asr_provider", "local") == "local":
            for key in ("asr_provider", "asr_qwen_model", "asr_window_seconds"):
                fields.pop(key, None)
        expected = canonical_hash(fields)
    elif version == 2:
        expected = extraction_hash(load_config(**settings))
    else:
        raise InputError("不支持的提取指纹版本")
    if expected != manifest.get("extraction_hash"):
        raise InputError("提取配置指纹不一致，请恢复原工作清单")
    return load_config(**{k: v for k, v in settings.items() if k not in obsolete})


def conflict_report(root: Path, message: str, suggestion: bytes = b"") -> Path:
    """写入独立冲突诊断和建议片段，返回其目录；不改基线、不登记成功版本。"""
    folder = contained(root, f".work/conflicts/{uuid.uuid4().hex}")
    folder.mkdir(parents=True, exist_ok=False)
    atomic_bytes(
        folder / "diagnostic.md",
        (
            "# 修订冲突\n\n" + message + "\n\n原版未改写，没有登记成功版本。"
            "请对照生成快照恢复唯一锚点，或手工合并建议片段后再运行。\n"
        ).encode("utf-8"),
    )
    atomic_bytes(
        folder / "suggestion.md",
        suggestion or ("未调用模型。无法确认目标范围，请先恢复锚点后重试。\n").encode(),
    )
    return folder


def load_baseline(root: Path, base: str) -> tuple[dict, Path, Notebook, bytes]:
    """读取显式成功版本，校验其目录、结构化索引和生成快照哈希。

    返回清单、版本目录、讲义对象和快照字节；当前 notes.md 可有手改，由修订流程另行核对。
    """
    if not re.fullmatch(r"r\d{3,}", base):
        raise InputError("基线版本格式为 r001、r002 等")
    manifest = read_json(contained(root, ".work/manifest.json"))
    if manifest.get("schema_version") != 1 or manifest.get("status") != "completed":
        raise InputError("此工作目录不是兼容的完整转换结果；请重新转换到独立目录")
    versions = manifest.get("versions", {})
    if (
        not isinstance(versions, dict)
        or not versions
        or any(
            not re.fullmatch(r"r\d{3,}", identity) or not isinstance(record, dict)
            for identity, record in versions.items()
        )
    ):
        raise InputError("版本清单结构损坏")
    if base not in versions or versions[base].get("status") != "completed":
        raise InputError("指定的成功基线版本不存在")
    folder = "." if base == "r001" else f"revisions/{base}"
    if versions[base].get("folder") != folder:
        raise InputError("版本清单路径不一致")
    directory = root if base == "r001" else contained(root, folder)
    notes = contained(directory, "notes.json")
    if digest(notes) != versions[base].get("notes_hash"):
        raise InputError("结构化索引已改变；只支持手改 notes.md，请恢复索引或重新转换")
    try:
        book = Notebook.model_validate(read_json(notes))
    except ValidationError:
        raise InputError("结构化索引版本不兼容或已损坏") from None
    snapshot = contained(root, f".work/versions/{base}.generated.md").read_bytes()
    if hashlib.sha256(snapshot).hexdigest() != versions[base].get("markdown_hash"):
        raise InputError("生成快照与清单不一致，停止复用")
    return manifest, directory, book, snapshot


def revise(
    workdir: Path,
    *,
    base: str = "r001",
    section: str | None = None,
    block: str | None = None,
    instruction: str | None = None,
    at_us: int | None = None,
    crop: tuple[int, int, int, int] | None = None,
    config_path: Path | None = None,
    secret: Path | None = None,
    allow_ai_additions: bool | None = None,
    progress: Callable[[str], None] | None = None,
    provider: Provider | None = None,
    model_provider: str | None = None,
    model: str | None = None,
) -> Path:
    """在显式基线上修订单章或单块，提交包含图片的新独立版本并返回目录。

    先校验快照、当前锚点、提取指纹及源素材；模型只接收目标范围的手改内容。
    按原始字节拼接非目标部分，发布前再次检查并发编辑和源变化，冲突保留诊断。
    精确换图直接使用本地媒体；文字修订才调用图文供应商。
    """
    if (section is None) == (block is None):
        raise InputError("须且只能选择 --section 或 --block")
    image_operation = at_us is not None or crop is not None
    if image_operation and (section or instruction):
        raise InputError("精确换图/裁剪只接受 --block，不能同时指定文字修订要求")
    if not image_operation and not (instruction and instruction.strip()):
        raise InputError("文字修订必须提供 --instruction")
    root = workdir.resolve()
    if not root.is_dir():
        raise InputError("工作目录不存在")
    # 先确认工作清单，再使用与 convert 相同的旁路锁，避免在任意输入目录创建工作数据。
    if not contained(root, ".work/manifest.json").is_file():
        raise InputError("此目录没有本工具的工作清单")
    with directory_lock(root.parent / f".{root.name}.lock"):
        manifest, baseline, book, snapshot = load_baseline(root, base)
        current_path = contained(baseline, "notes.md")
        if current_path.stat().st_size > 30_000_000:
            raise InputError("当前 Markdown 超过 30 MB，不能安全进行局部修订")
        current = current_path.read_bytes()
        current_hash = hashlib.sha256(current).hexdigest()
        target_id = section or block
        try:
            spans = expected_spans(current, book)
            original_spans = expected_spans(snapshot, book)
            if target_id not in spans:
                raise InputError("目标 ID 不存在，请查看当前文档中的章节/块 ID")
        except AnchorConflict as exc:
            report = conflict_report(root, str(exc))
            raise InputError(f"{exc}；诊断与建议片段：{report}") from exc
        span = spans[target_id]
        target_chapter = next(c for c in book.chapters if c.id == (section or span.parent))
        target_block = next((b for b in target_chapter.blocks if b.id == block), None)
        if image_operation and (target_block is None or target_block.kind != "figure"):
            raise InputError("指定块不是图片块")
        # 先核对手加图片依赖，避免已知无法导出的请求仍然产生模型费用。
        for reference in image_dependencies(current):
            image = contained(baseline, local_image(reference))
            if not image.is_file():
                raise InputError("基线文档中的图片依赖缺失")
        original_config = baseline_config(manifest)
        settings = original_config.model_dump()
        if config_path:
            import tomllib

            try:
                settings = merge_provider_settings(
                    settings, tomllib.loads(config_path.read_text(encoding="utf-8"))
                )
            except (OSError, ValueError):
                raise InputError("修订配置 TOML 无效") from None
        settings = merge_provider_settings(settings, {"provider": model_provider, "model": model})
        if instruction is not None:
            settings["instruction"] = instruction
        if secret is not None:
            settings["secret_file"] = str(secret)
        if allow_ai_additions is not None:
            settings["allow_ai_additions"] = allow_ai_additions
        config = load_config(**settings)
        if extraction_hash(config) != extraction_hash(original_config):
            # 模型可换，提取设置不能换；当前修订必须继续使用基线对应的同一组证据。
            raise InputError("提取配置与基线不一致，请重新转换到独立目录")
        local_source = read_json(contained(root, ".work/source-local.json"))
        if canonical_hash(local_source.get("files")) != manifest.get("source_hash"):
            raise InputError("本地来源清单与原始指纹不一致")
        source_path = Path(local_source["path"])
        source = inspect_source(source_path)
        subtitle = Path(local_source["subtitle"]) if local_source.get("subtitle") else None
        fingerprints = fingerprint_source(source_path, source, subtitle)
        if fingerprints != local_source["files"] or source != book.source:
            raise InputError("源素材与基线不一致，请重新转换到独立目录")
        validate_notebook(book, root)
        if current != snapshot:
            # 不把保留的手改反向猜写进索引，而是标明两者尚未核验同步。
            book.sync_status = "manual_unverified"
        for chapter in book.chapters:
            for note_block in chapter.blocks:
                now = spans[note_block.id]
                previous = original_spans[note_block.id]
                if current[now.start : now.end] != snapshot[previous.start : previous.end]:
                    note_block.sync_status = "manual_unverified"
        number = max(int(identity[1:]) for identity in manifest["versions"]) + 1
        revision_id = f"r{number:03d}"
        destination = contained(root, f"revisions/{revision_id}")
        if destination.exists():
            raise InputError("新版本目录已存在但未登记，可能有未完成提交，请保留并人工核对")
        staging = contained(root, f".work/revision-{uuid.uuid4().hex}.partial")
        staging.mkdir(parents=True, exist_ok=False)
        events = Events(root, progress)
        replacement = b""
        try:
            with events.stage(f"revise:{revision_id}"):
                if image_operation:
                    old_frame = frame_of(book, target_block.frame_id)
                    requested = old_frame.at_us if at_us is None else at_us
                    selected_crop = old_frame.crop if crop is None else crop
                    if not book.start_us <= requested < book.end_us:
                        raise InputError("替换帧须在当前转换范围内；其他区间请独立转换")
                    new_frame = register_frame(
                        source_path,
                        source,
                        requested,
                        book.end_us,
                        root,
                        f"frame-{revision_id}-{target_block.id}",
                        selected_crop,
                        parent_id=old_frame.id,
                    )
                    book.frames.append(new_frame)
                    target_block.frame_id = new_frame.id
                    target_block.evidence_ids = [
                        new_frame.id if identity == old_frame.id else identity
                        for identity in target_block.evidence_ids
                    ]
                    # 保留当前手改图注，只替换受控图片和来源行；无法确定结构时报告冲突。
                    target_current = current[span.start : span.end]
                    target_original = snapshot[
                        original_spans[target_id].start : original_spans[target_id].end
                    ]
                    replacement = replace_figure(
                        target_current, target_original, target_block, book
                    )
                    book.review.append(
                        ReviewItem(
                            reason="已按指定帧/裁剪更新图片；原图注及相关文字保留，请核对图文一致性",
                            start_us=target_chapter.start_us,
                            end_us=target_chapter.end_us,
                            block_id=target_block.id,
                            evidence_ids=[new_frame.id],
                        )
                    )
                else:
                    active_provider = provider or create_provider(config, events)
                    fixed_chapter = target_chapter.model_copy(deep=True)
                    if target_block:
                        from .documents import evidence_range

                        ranges = [
                            evidence_range(book, identity) for identity in target_block.evidence_ids
                        ]
                        if ranges:
                            fixed_chapter.start_us = max(book.start_us, min(r[0] for r in ranges))
                            fixed_chapter.end_us = min(book.end_us, max(r[1] for r in ranges) + 1)
                    packet, images = evidence_packet(
                        book,
                        fixed_chapter,
                        config,
                        root,
                        current_markdown=current[span.start : span.end].decode("utf-8"),
                        target_ids=[target_id],
                    )
                    if target_block and target_block.kind == "figure":
                        # 图注修订固定原图，不能让模型借此执行 P1 的自然语言重新选图。
                        pinned = frame_of(book, target_block.frame_id)
                        packet["frames"] = [
                            {"id": pinned.id, "at_us": pinned.at_us, "crop": pinned.crop}
                        ]
                        images = [(pinned.id, contained(root, pinned.path))]
                    draft = active_provider.compose(packet, images)
                    validate_draft(draft, packet)
                    if target_block:
                        if len(draft.blocks) != 1 or draft.blocks[0].kind != target_block.kind:
                            raise TaskError("单块修订必须返回一个相同类型的块")
                        updated = NoteBlock(
                            **draft.blocks[0].model_dump(),
                            id=target_block.id,
                            chapter_id=target_chapter.id,
                        )
                        target_chapter.blocks = [
                            updated if b.id == updated.id else b for b in target_chapter.blocks
                        ]
                        replacement = render_block(updated, book).rstrip(b"\n") + b"\n"
                    else:
                        old_blocks = target_chapter.blocks
                        used = set()
                        new_blocks = []
                        for index, draft_block in enumerate(draft.blocks, 1):
                            # 优先复用同类型旧 ID；超出旧块数量时加入版本号，避免新旧 ID 冲突。
                            candidate = next(
                                (
                                    b
                                    for b in old_blocks
                                    if b.kind == draft_block.kind and b.id not in used
                                ),
                                None,
                            )
                            prefix = "fig" if draft_block.kind == "figure" else "blk"
                            identity = (
                                candidate.id
                                if candidate
                                else f"{prefix}-{target_chapter.id[3:]}-{revision_id}-{index:03d}"
                            )
                            used.add(identity)
                            new_blocks.append(
                                NoteBlock(
                                    **draft_block.model_dump(),
                                    id=identity,
                                    chapter_id=target_chapter.id,
                                )
                            )
                        target_chapter.blocks = new_blocks
                        # 保留原章节标题，使目标外的目录也无需改写。
                        replacement = render_chapter(target_chapter, book).rstrip(b"\n") + b"\n"
                    for reason in draft.review:
                        book.review.append(
                            ReviewItem(
                                reason=reason,
                                start_us=target_chapter.start_us,
                                end_us=target_chapter.end_us,
                                block_id=target_id,
                            )
                        )
                new_current = current[: span.start] + replacement + current[span.end :]
                validate_notebook(book, root)
                expected_spans(new_current, book)
                stage_assets(book, root, staging, baseline)
                copy_dependencies(new_current, baseline, staging)
                export_book(book, root, staging, new_current)
                request_summary = instruction or f"指定帧/裁剪 {at_us} / {crop}"
                atomic_bytes(
                    contained(staging, "changes.md"),
                    (
                        f"# {revision_id} 修改摘要\n\n- 基线：{base}\n- 目标：{target_id}\n"
                        f"- 基线当前文稿 SHA-256：{current_hash}\n"
                        f"- 要求：{request_summary}\n"
                        "- 只替换目标范围；非目标 Markdown 字节保持不变。\n"
                        "- 关联证据及待核对事项见 notes.json 与 review.md。\n"
                    ).encode(),
                )
                if digest(current_path) != current_hash:
                    # 独占锁约束工具自身，但不能阻止用户用编辑器改文稿，提交前必须复核。
                    report = conflict_report(root, "模型请求或导出期间基线被再次编辑", replacement)
                    raise InputError(f"检测到并发编辑；建议片段：{report}")
                if fingerprint_source(source_path, source, subtitle) != fingerprints:
                    raise InputError("修订期间源素材发生变化，未发布版本")
                destination.parent.mkdir(parents=True, exist_ok=True)
                staging.rename(destination)
                # 完整目录先落位，再写快照与清单；中途失败的目录不能自动视为已登记版本。
                atomic_bytes(
                    contained(root, f".work/versions/{revision_id}.generated.md"), new_current
                )
                manifest["versions"][revision_id] = {
                    "folder": f"revisions/{revision_id}",
                    "base": base,
                    "status": "completed",
                    "markdown_hash": hashlib.sha256(new_current).hexdigest(),
                    "notes_hash": digest(contained(destination, "notes.json")),
                    "base_current_hash": current_hash,
                    "target": target_id,
                }
                write_json(contained(root, ".work/manifest.json"), manifest)
            return destination
        except AnchorConflict as exc:
            report = conflict_report(root, str(exc), replacement)
            raise InputError(f"{exc}；建议片段：{report}") from exc
        except BaseException as exc:
            # 失败暂存目录只供诊断，不能登记为成功修订。
            if staging.exists():
                write_json(
                    contained(staging, "failure.json"),
                    {"status": "failed", "error_type": type(exc).__name__},
                )
            raise


def replace_figure(current: bytes, original: bytes, block: NoteBlock, book: Notebook) -> bytes:
    """仅替换渲染器拥有的图片行和来源行，逐字节保留手改图注及原有换行格式。

    受控行必须仍与生成快照一致；缺失、重复或手改均报告冲突，不猜测用户意图。
    """
    current_lines = current.splitlines(keepends=True)
    original_lines = original.splitlines(keepends=True)
    rendered_lines = render_block(block, book).splitlines(keepends=True)
    for prefix in (b"![", "> 来源：".encode()):
        old = [line for line in original_lines if line.startswith(prefix)]
        new = [line for line in rendered_lines if line.startswith(prefix)]
        existing = [i for i, line in enumerate(current_lines) if line.startswith(prefix)]
        if len(old) != 1 or len(new) != 1 or len(existing) != 1:
            raise AnchorConflict("图片块的受控图片/来源行缺失或重复")
        index = existing[0]
        if current_lines[index].rstrip(b"\r\n") != old[0].rstrip(b"\r\n"):
            raise AnchorConflict("图片块的受控图片/来源行已有手改，不能自动替换；请手工合并建议")
        ending = b"\r\n" if current_lines[index].endswith(b"\r\n") else b"\n"
        current_lines[index] = new[0].rstrip(b"\r\n") + ending
    return b"".join(current_lines)
