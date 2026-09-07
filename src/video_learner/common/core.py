"""与 CLI 无关的错误、时间和路径契约。"""

import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

US = 1_000_000


class InputError(ValueError):
    """输入、配置或不安全的文件操作。"""


class TaskError(RuntimeError):
    """运行失败，不表示输入语法有误。"""


def parse_time(value: str) -> int:
    """将秒数或冒号分隔时间转成非负整数微秒。

    使用 Decimal 保留最多六位小数，避免浮点误差改变片段边界；非法时间抛 InputError。
    """
    try:
        if not re.fullmatch(r"\d+(?::\d{2}){0,2}(?:\.\d{1,6})?", value):
            raise ValueError
        parts = [Decimal(part) for part in value.split(":")]
        if len(parts) > 1 and any(part >= 60 for part in parts[1:]):
            raise ValueError
        seconds = Decimal(0)
        for part in parts:
            seconds = seconds * 60 + part
        return int(seconds * US)
    except (InvalidOperation, ValueError) as exc:
        raise InputError("时间须为非负秒数或 HH:MM:SS，最多六位小数") from exc


def timestamp(value: int) -> str:
    """将微秒向下取整到秒，生成展示用时间；内部计算仍使用原始微秒。"""
    seconds = value // US
    return f"{seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}"


def time_range(start: int, end: int | None, duration: int) -> tuple[int, int]:
    """补齐省略的结束时间，并验证原视频时间线上的非空半开区间 [start, end)。"""
    stop = duration if end is None else end
    if not 0 <= start < stop <= duration:
        raise InputError(f"时间范围必须满足 0 <= start < end <= {timestamp(duration)}")
    return start, stop


def contained(root: Path, relative: str | Path) -> Path:
    """解析目录内的相对资源路径，拒绝绝对路径、上级跳转和解析后的越界目标。

    resolve 会展开符号链接及 Windows 目录联接，不能只靠字符串前缀判断目录归属。
    """
    base = root.resolve()
    name = Path(relative)
    if name.is_absolute() or name.drive or ".." in name.parts:
        raise InputError("资源路径必须是目录内的相对路径")
    target = (base / name).resolve()
    if not target.is_relative_to(base) or target == base:
        raise InputError("资源路径越出允许目录")
    return target


def output_path(source: Path, output: Path) -> Path:
    """返回新的绝对输出目录，拒绝与素材目录相同或互为祖先的路径。

    单文件输入也保护其整个父目录，防止生成产物混入素材或覆盖已有输出。
    """
    try:
        source = source.resolve(strict=True)
    except OSError as exc:
        raise InputError("素材路径不存在或不可读") from exc
    root = source if source.is_dir() else source.parent
    target = output.resolve()
    if target == root or target.is_relative_to(root) or root.is_relative_to(target):
        raise InputError("输出目录不能与源目录重叠")
    if target.exists():
        raise InputError("输出目录已存在；请指定新的独立目录")
    return target
