"""验证实际 CLI 入口的帮助输出和参数错误。"""

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("args", "exit_code", "expected"),
    [
        ([], 0, "检查本地视频或单课时缓存"),
        (["unknown-command"], 2, "参数错误："),
    ],
)
def test_cli_help_and_argument_errors(args, exit_code, expected):
    """经真实入口检查退出码，避免仅测 Typer 应用而漏掉入口追加的错误提示。"""
    result = subprocess.run(
        [sys.executable, "-m", "video_learner.cli", *args],
        capture_output=True,
        encoding="utf-8",
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == exit_code
    assert expected in output
    if exit_code == 0:
        assert "参数错误" not in output
        assert not result.stderr
    else:
        assert result.stderr.partition("参数错误：")[2].strip()
