"""按职能组织审阅规则；供应商只负责发送请求。"""

from video_learner.common.config import ReviewStep
from video_learner.notes.prompts import (
    EVIDENCE_RULES,
    EXAMPLES,
    FIDELITY_RULES,
    FIGURE_RULES,
    REVIEW_RULES,
)

COMMON = "\n".join(
    (
        "你是独立的中文讲义审阅者，对照当前讲义与原课证据，返回 ReviewResult JSON。",
        EVIDENCE_RULES,
        FIDELITY_RULES,
        REVIEW_RULES,
        EXAMPLES,
        r"""【审阅与输出】
current_markdown 是用户当前文稿，优先于 chapter 中可能尚未同步的 body；保留手改意图。
有明确证据支持时局部纠正；证据不足时用 action=report，说明具体断言与需回看确认的内容。
文字可以引用本章图片，图文互补不要求语音逐字复述画面。正常内容保留，没有问题时 findings=[]。
findings 包含原块 target_id、kind、具体 reason、本章 evidence_ids、action 和 blocks。
replace 用 blocks 替换原块，空列表删除；insert_after 插在该块后；report 仅登记疑点，blocks=[]。
同一块最多一次修改。正文使用普通 Markdown，内部 ID 只填结构字段；路径、时间戳和锚点
由程序生成，原课代码中的路径与时间常量可保留。正文禁止 ![](...) 图片语法，LaTeX 正确转义。
current_review 每项有 id；应用模式下，证据确认已解决的项或符合上述排除范围的无效提示，
将 id 放入 resolved_review_ids，未提及的疑点保留；新增 report 与已有问题避免重复。
""",
    )
)

VISUAL = """本轮职能：图文对应。检查配图的教学用途、遗漏、重复、位置和图注。
transcript_ids 填真正相关的本章语音，纯图知识可为空；speech_window_ids 仅表示时间邻近。
正文的问题留给内容审阅，本轮可以报告具体问题。
"""

VISUAL_APPLY = """【图文应用契约】
frames 覆盖每一张候选图且仅出现一次：decision=use 表示最终显示，omit 表示不选用。
reason 简述具体用途或省略原因，选图理由只放 frames，不写入 findings 的 report。
related_block_id 指向保留的原文字块表示放在其后，指向原图片块表示沿用那个位置。
程序按 frames 决定增删移动图片，findings 只允许修改已有图注或报告问题，不改正文。
纠正图注用 replace，返回一个同 frame_id 的 figure 块；无需图注时 body 为空字符串。
"""

CONTENT = """本轮职能：讲义内容。对照证据检查对象、条件、术语、代码符号、原文引语、
材料与评价归属、关键步骤和逻辑方向；纠正文体时保留原有信息与推理。
检查跨章半句、重复论述、生成过程说明、内部证据 ID、空标题和堆砌界面字段。
在证据支持的完整观点处收束，后章已续接时不报告假缺失。图片或图注的问题仅报告。
"""

CONTENT_APPLY = """【内容应用契约】
frames=null，保留图片的位置、选择和图注。findings 只修改文字块，替换和插入只返回 text 块。
"""


def review_prompt(step: ReviewStep) -> str:
    """组合固定证据边界、职能和用户规则；报告模式禁止应用任何修改。"""
    prompt = COMMON + (VISUAL + FIGURE_RULES if step.kind == "visual" else CONTENT)
    if step.instruction:
        prompt += "\n用户自定义审阅要求（在上述职能与证据范围内执行）：\n" + step.instruction
    if step.mode == "report":
        prompt += (
            "\n本轮为只报告模式：frames=null，findings 只能用 report，"
            "resolved_review_ids=[]。只登记具体问题，保持讲义和已有疑点原样。"
        )
    else:
        prompt += VISUAL_APPLY if step.kind == "visual" else CONTENT_APPLY
    return prompt
