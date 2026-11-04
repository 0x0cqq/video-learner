"""共用工作目录与版本规则：指纹、显式基线、手改保护和新版本提交。"""

import hashlib
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from video_learner.common.config import Config, load_config
from video_learner.common.core import InputError, contained
from video_learner.common.schemas import Notebook, Source
from video_learner.common.storage import atomic_bytes, canonical_hash, digest, read_json, write_json
from video_learner.media.io import inspect_source, source_file
from video_learner.notes.rendering import (
    AnchorConflict,
    Span,
    expected_spans,
    image_dependencies,
    local_image,
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
    if path.is_dir() and (path / "download.json").is_file():
        files.add((path / "download.json").resolve())
    return {str(file): digest(file) for file in sorted(files)}


def baseline_config(manifest: dict) -> Config:
    """加载已保存版本的设置并核对提取指纹；读取旧上下文字段时统一其名称。"""
    if manifest.get("extraction_version") != 4:
        raise InputError("提取数据版本不兼容，请重新转换到独立目录")
    settings = manifest.get("config")
    if not isinstance(settings, dict):
        raise InputError("工作清单配置无效")
    config = load_config(base=settings)
    if extraction_hash(config) != manifest.get("extraction_hash"):
        raise InputError("提取配置指纹不一致，请恢复原工作清单")
    return config


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
    if manifest.get("schema_version") != 2 or manifest.get("status") != "completed":
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


@dataclass(frozen=True)
class EditableBaseline:
    """显式版本的当前字节、原始快照和定位结果；book 记录保留的手改状态。"""

    base: str
    manifest: dict
    directory: Path
    book: Notebook
    current: bytes
    current_hash: str
    snapshot: bytes
    spans: dict[str, Span]
    original_spans: dict[str, Span]


@dataclass(frozen=True)
class VerifiedSource:
    """与基线匹配的本机素材及当次指纹，供读取证据和提交前复核。"""

    path: Path
    source: Source
    subtitle: Path | None
    fingerprints: dict


def load_editable_baseline(root: Path, base: str) -> EditableBaseline:
    """核验当前锚点和图片依赖，标记手改；锚点冲突只写独立诊断。"""
    manifest, directory, book, snapshot = load_baseline(root, base)
    current_path = contained(directory, "notes.md")
    if current_path.stat().st_size > 30_000_000:
        raise InputError("当前 Markdown 超过 30 MB，不能安全进行局部修订")
    current = current_path.read_bytes()
    try:
        spans = expected_spans(current, book)
        original_spans = expected_spans(snapshot, book)
    except AnchorConflict as exc:
        report = conflict_report(root, str(exc))
        raise InputError(f"{exc}；诊断与建议片段：{report}") from exc
    for reference in image_dependencies(current):
        if not contained(directory, local_image(reference)).is_file():
            raise InputError("基线文档中的图片依赖缺失")
    if current != snapshot:
        book.sync_status = "manual_unverified"
    for chapter in book.chapters:
        for block in chapter.blocks:
            now, old = spans[block.id], original_spans[block.id]
            if current[now.start : now.end] != snapshot[old.start : old.end]:
                block.sync_status = "manual_unverified"
    return EditableBaseline(
        base,
        manifest,
        directory,
        book,
        current,
        hashlib.sha256(current).hexdigest(),
        snapshot,
        spans,
        original_spans,
    )


def verify_source(root: Path, manifest: dict, book: Notebook) -> VerifiedSource:
    """核对本机来源清单、实际文件和媒体结构；展示标题变化保持证据身份。"""
    local = read_json(contained(root, ".work/source-local.json"))
    if canonical_hash(local.get("files")) != manifest.get("source_hash"):
        raise InputError("本地来源清单与原始指纹不一致")
    path = Path(local["path"])
    subtitle = Path(local["subtitle"]) if local.get("subtitle") else None
    source = inspect_source(path)
    fingerprints = fingerprint_source(path, source, subtitle)
    if fingerprints != local["files"] or source.model_dump(
        exclude={"title"}
    ) != book.source.model_dump(exclude={"title"}):
        raise InputError("源素材与基线不一致，请重新转换到独立目录")
    return VerifiedSource(path, source, subtitle, fingerprints)


def next_revision(root: Path, manifest: dict) -> tuple[str, Path]:
    """分配下一个版本号，并在模型请求前拒绝未登记的同名目录。"""
    number = max(int(identity[1:]) for identity in manifest["versions"]) + 1
    identity = f"r{number:03d}"
    destination = contained(root, f"revisions/{identity}")
    if destination.exists():
        raise InputError("新版本目录已存在但未登记，请保留并人工核对")
    return identity, destination


def commit_revision(
    root: Path,
    baseline: EditableBaseline,
    source: VerifiedSource,
    staging: Path,
    destination: Path,
    current: bytes,
    details: dict,
    suggestion: bytes,
) -> Path:
    """调用方持锁并完成导出后，复核输入、发布目录，再保存快照与成功版本记录。

    中途失败的目录保持未登记状态；基线并发编辑时写建议片段，不覆盖当前文稿。
    """
    if digest(contained(baseline.directory, "notes.md")) != baseline.current_hash:
        report = conflict_report(root, "处理期间基线被再次编辑", suggestion)
        raise InputError(f"检测到并发编辑；建议片段：{report}")
    if fingerprint_source(source.path, source.source, source.subtitle) != source.fingerprints:
        raise InputError("处理期间源素材发生变化，未发布版本")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(destination)
    identity = destination.name
    atomic_bytes(contained(root, f".work/versions/{identity}.generated.md"), current)
    baseline.manifest["versions"][identity] = {
        "folder": f"revisions/{identity}",
        "base": baseline.base,
        "status": "completed",
        "markdown_hash": hashlib.sha256(current).hexdigest(),
        "notes_hash": digest(contained(destination, "notes.json")),
        "base_current_hash": baseline.current_hash,
        **details,
    }
    write_json(contained(root, ".work/manifest.json"), baseline.manifest)
    return destination
