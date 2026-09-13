"""可用本地确定性处理修复的阅读问题，不依赖模型评审。"""

from video_learner.notes.rendering import normalize_headings


def test_heading_normalization_preserves_code_and_nested_blocks():
    """正文层级归入章节，代码中的同形井号、引用及 Setext 内容保持语义。"""
    body = "## 子题\n\n```python\n# 注释\n## 保留\n```\n\n> # 引用子题\n\n小题\n---\n"
    normalized = normalize_headings(body)
    assert normalized == (
        "### 子题\n\n```python\n# 注释\n## 保留\n```\n\n> ### 引用子题\n\n### 小题\n"
    )
    assert normalize_headings(normalized) == normalized
