"""讲义正文的确定性质量约束；语义疑点由独立复审对照证据判断。"""

import re
from collections.abc import Iterable

from video_learner.common.core import TaskError
from video_learner.common.schemas import Draft
from video_learner.notes.rendering import MARKDOWN


def validate_editorial_structure(draft: Draft, evidence_ids: Iterable[str]) -> None:
    """拒绝正文泄漏内部证据 ID 和悬空末标题，保留代码块里的字面示例。

    不删改关键词或猜测半句话，失败交给有界修复；旧冻结草稿由原契约重放。
    """
    identifiers = set(evidence_ids)
    prose = [draft.title]
    for block in draft.blocks:
        for token in MARKDOWN.parse(block.body):
            if token.type == "inline":
                prose.extend(
                    child.content
                    for child in token.children or []
                    if child.type in ("text", "code_inline")
                )
    for text in prose:
        if any(word in identifiers for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text)):
            raise TaskError("正文含内部证据 ID；引用放在 evidence_ids，正文直接讲解内容")
    # 图片也是标题的有效内容，不能把仅有图而没有图注的小节误报为空。
    content = "\n\n".join(
        block.body if block.kind == "text" else "图片\n\n" + block.body for block in draft.blocks
    )
    tokens = MARKDOWN.parse(content)
    if tokens and tokens[-1].type == "heading_close":
        raise TaskError("章节末尾只有标题，没有正文或图片；请完成该小节或移除空标题")
