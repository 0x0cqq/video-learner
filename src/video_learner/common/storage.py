"""有界文件访问、原子提交与操作系统持有的目录锁。"""

import hashlib
import json
import os
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from video_learner.common.core import InputError, contained


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical_hash(value: object) -> str:
    """对按键排序的 UTF-8 JSON 求哈希，使配置字典的插入顺序不影响指纹。"""
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def atomic_bytes(path: Path, content: bytes) -> None:
    """先写同目录临时文件并 fsync，再原子替换目标；异常时清理临时文件。

    此函数允许替换已有文件，不负责版本保护或加锁，调用方须先确定可写目标。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: object) -> None:
    """将完整 JSON 编码并回读验证后原子写入，避免留下只写了一部分的清单。"""
    payload = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    json.loads(payload)
    atomic_bytes(path, payload)


def read_json(path: Path) -> dict:
    """有界读取 UTF-8 JSON 对象；缺失、超大或结构损坏统一报告为输入错误。"""
    try:
        if path.stat().st_size > 50_000_000:
            raise ValueError("too large")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("expected object")
        return value
    except (OSError, ValueError) as exc:
        raise InputError("工作清单或结构化文档缺失/损坏，请重新转换到独立目录") from exc


@contextmanager
def directory_lock(path: Path):
    """锁由操作系统释放；保留锁文件，避免 unlink 后两个 inode 同时上锁。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise InputError("同一输出目录已有任务正在写入") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class Events:
    def __init__(self, root: Path, progress: Callable[[str], None] | None = None):
        """为本次操作建立独立运行 ID 和单调时钟，追加日志而不改写已有记录。"""
        self.path = contained(root, ".work/logs/events.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.progress = progress or (lambda _: None)
        self.run_id = uuid.uuid4().hex
        self.started = time.monotonic()
        self.usage_events: list[dict] = []

    def emit(self, stage: str, status: str, **details) -> None:
        """追加带 UTC 与运行内相对时间的事件；details 须由调用方筛除凭据和原始异常。"""
        record = {
            "stage": stage,
            "status": status,
            **details,
            "run_id": self.run_id,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.monotonic() - self.started,
        }
        if stage in ("model_call", "asr_model_call", "model_usage", "asr_model_usage"):
            if status in ("running", "received"):
                self.usage_events.append(record)
        # 这里只序列化受控字段，不负责从供应商原始错误或 URL 中自动剔除敏感值。
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.progress(
            f"{stage}：{status}" + (f" ({details['seconds']:.1f}s)" if "seconds" in details else "")
        )

    @contextmanager
    def stage(self, name: str):
        """记录阶段耗时和完成、失败或中断状态，并将异常原样交还应用层。"""
        started = time.monotonic()
        self.emit(name, "running")
        try:
            yield
        except BaseException as exc:
            self.emit(
                name,
                "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
                seconds=time.monotonic() - started,
                error_type=type(exc).__name__,
            )
            raise
        else:
            self.emit(name, "completed", seconds=time.monotonic() - started)
