"""讲义模型交互：统一管理任务规则、历史上下文、校验修复与调用预算。"""

import base64
import copy
import json
import time
import uuid
from pathlib import Path
from typing import Literal, Protocol

from pydantic import ValidationError

from video_learner.common.config import Config, ReviewStep
from video_learner.common.core import InputError, TaskError, contained
from video_learner.common.schemas import Draft, ReviewResult
from video_learner.common.storage import Events, atomic_bytes
from video_learner.notes.prompts import GENERATION_TASK, PROMPT_VERSION, SYSTEM_PROMPT
from video_learner.notes.review_prompts import review_prompt
from video_learner.providers.base import ModelRequest, Provider, ProviderError, create_provider


class Generator(Protocol):
    def compose(self, packet: dict, images: list[tuple[str, Path]]) -> Draft: ...

    def review(self, packet: dict, images: list[tuple[str, Path]]) -> ReviewResult: ...


def strict_schema(
    response_type: type[Draft] | type[ReviewResult] = Draft,
    step: ReviewStep | None = None,
    target_kind: Literal["text", "figure"] | None = None,
) -> dict:
    """派生响应 schema，约束块类型、frame_id 及本轮复审允许的操作。

    必填字段与额外字段由数据模型定义；复审 schema 提前排除越权输出，本地仍校验目标与证据。
    """
    schema = response_type.model_json_schema()
    block_schema = schema["$defs"]["DraftBlock"]
    alternatives = []
    kinds = (
        ((target_kind,) if target_kind is not None else ("text", "figure"))
        if step is None
        else (("figure",) if step.kind == "visual" else ("text",))
    )
    for kind in kinds:
        alternative = copy.deepcopy(block_schema)
        alternative["properties"]["kind"] = {"type": "string", "const": kind}
        alternative["properties"]["frame_id"] = (
            {"type": "null"} if kind == "text" else {"type": "string", "minLength": 1}
        )
        if kind == "text":
            alternative["properties"]["body"]["minLength"] = 1
        alternatives.append(alternative)
    schema["$defs"]["DraftBlock"] = {"anyOf": alternatives}
    if target_kind is not None:
        schema["properties"]["blocks"]["maxItems"] = 1
    if step is not None:
        if step.mode == "report" or step.kind == "content":
            schema["properties"]["frames"] = {"type": "null"}
        else:
            schema["properties"]["frames"] = {
                "type": "array",
                "items": {"$ref": "#/$defs/FrameAssessment"},
                "maxItems": 30,
            }
        if step.mode == "report":
            finding = schema["$defs"]["ReviewFinding"]["properties"]
            finding["action"] = {"type": "string", "const": "report"}
            finding["blocks"]["maxItems"] = 0
            schema["properties"]["resolved_review_ids"]["maxItems"] = 0

    return schema


class NoteGenerator:
    def __init__(
        self, config: Config, events: Events, provider: Provider, *, response_directory: Path
    ):
        """为一次业务运行建立会话，显式指定响应目录；转换和复审共用调用预算。"""
        self.config, self.events, self.provider, self.calls = config, events, provider, 0
        self.response_directory = response_directory.resolve()
        self._history: list[dict] = []
        self._request_prefix: list[dict] = []
        self._last_input: list[dict] = []

    def _prepare_history(self, content: list[dict], packet: dict) -> None:
        """完整复用已成功请求前缀；到预算边界整组重置，由包内上章摘要继续衔接。

        文本以 UTF-8 字节数作保守 token 估计，每图按本地预算单位估计 1024 token；
        请求体单独按 JSON 字节限制。预留 schema、结构、修复提示和输出空间。
        """
        enabled = (
            self.config.composition_context == "history" and packet.get("operation") == "convert"
        )
        self._request_prefix = self._history if enabled else []
        current = [{"role": "user", "content": content}]

        def measure(messages: list[dict]) -> tuple[int, int]:
            """只估算上下文与传输预算；计费始终使用服务端实际 usage。"""
            tokens = 0
            for message in messages:
                parts = message["content"]
                if isinstance(parts, str):
                    tokens += len(parts.encode("utf-8"))
                else:
                    tokens += sum(
                        1024 if p["type"] == "image_url" else len(p["text"].encode("utf-8"))
                        for p in parts
                    )
            overhead = (
                len(self._system_prompt.encode("utf-8")) + len(json.dumps(self._schema)) * 2 + 4096
            )
            return (
                tokens + overhead + self.config.max_output_tokens,
                len(json.dumps(messages, ensure_ascii=False).encode("utf-8")) + overhead,
            )

        tokens, size = measure(self._request_prefix + current)
        if (
            tokens > self.config.context_token_budget
            or size > self.config.context_max_megabytes * 1024**2
        ):
            self.events.emit("model_context", "reset", reason="budget")
            self._history = []
            self._request_prefix = []
            tokens, size = measure(current)
        if (
            tokens > self.config.context_token_budget
            or size > self.config.context_max_megabytes * 1024**2
        ):
            raise TaskError("当前章节超过上下文或请求体预算，请缩短章节或减少候选图")
        self.events.emit(
            "model_context",
            "prepared",
            history_turns=len(self._request_prefix) // 2,
            estimated_tokens=tokens,
            request_bytes=size,
        )

    def compose(self, packet: dict, images: list[tuple[str, Path]]) -> Draft:
        """组织正文；请求结果经结构、证据与确定性文体约束后才能采用。"""
        return self._generate(packet, images, Draft)

    def review(self, packet: dict, images: list[tuple[str, Path]]) -> ReviewResult:
        """独立复审，不继承生成会话；共用实际调用预算、用量和有界修复。"""
        return self._generate(packet, images, ReviewResult)

    def _generate[T: (Draft, ReviewResult)](
        self, packet: dict, images: list[tuple[str, Path]], response_type: type[T]
    ) -> T:
        """以有界文本和图片证据请求草稿，保存最终响应并执行结构及语义约束校验。

        网络重试与内容修复共用实际调用预算；认证错误和未完成响应直接终止。
        供应商只执行单次请求；历史、任务规则、修复和实际调用预算由本层统一管理。
        """
        step = (
            ReviewStep.model_validate(packet["review_step"])
            if response_type is ReviewResult
            else None
        )
        self._schema = strict_schema(response_type, step, packet.get("target_block_kind"))
        self._system_prompt = review_prompt(step) if step is not None else SYSTEM_PROMPT
        payload_started = time.monotonic()
        text = json.dumps(packet, ensure_ascii=False)
        if len(text.encode("utf-8")) > 250_000 or len(images) > self.config.max_images_per_chapter:
            raise TaskError("单次证据包超过大小限制，请缩短章节或减少采样")
        content = [{"type": "text", "text": text}]
        total_image_bytes = 0
        for identity, path in images:
            total_image_bytes += path.stat().st_size
            if total_image_bytes > 25_000_000:
                raise TaskError("单次图像证据超过 25 MB，请减少候选图数量")
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.extend(
                [
                    {
                        "type": "text",
                        "text": "图像证据："
                        + json.dumps(
                            next(item for item in packet["frames"] if item["id"] == identity),
                            ensure_ascii=False,
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}", "detail": "high"},
                    },
                ]
            )
        if response_type is Draft:
            content.append({"type": "text", "text": GENERATION_TASK})
        self.events.emit(
            "model_payload",
            "prepared",
            seconds=time.monotonic() - payload_started,
            text_bytes=len(text.encode("utf-8")),
            image_bytes=total_image_bytes,
            image_count=len(images),
        )
        repair = None
        self._prepare_history(content, packet)
        for attempt in range(self.config.max_retries + 1):
            if self.calls >= self.config.max_calls:
                raise TaskError("达到模型调用次数上限，剩余章节未完成")
            self.calls += 1
            started = time.monotonic()
            self.events.emit(
                "model_call",
                "running",
                call=self.calls,
                model=self.config.model,
                attempt=attempt + 1,
                chapter_id=packet.get("chapter_id"),
                operation=packet.get("operation"),
                review_step=packet.get("review_step", {}).get("name"),
            )
            try:
                self._last_input = self._request_prefix + [
                    {
                        "role": "user",
                        "content": content + ([{"type": "text", "text": repair}] if repair else []),
                    }
                ]
                response = self.provider.generate(
                    ModelRequest(self._system_prompt, self._schema, self._last_input),
                    call=self.calls,
                )
                self.events.emit(
                    "model_usage",
                    "received",
                    call=self.calls,
                    model=self.config.model,
                    prompt_version=PROMPT_VERSION,
                    seconds=time.monotonic() - started,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    cached_input_tokens=response.cached_input_tokens,
                )
                if not response.completed:
                    self.events.emit("model_call", "incomplete", call=self.calls)
                    raise TaskError("模型输出未完成，可能达到输出限制")
                raw_path = contained(self.response_directory, f"{uuid.uuid4().hex}.json")
                # 完整响应即使校验失败也保留供诊断；未完成响应不会走到这里。
                atomic_bytes(raw_path, response.text.encode("utf-8"))
                self.events.emit(
                    "model_response", "saved", call=self.calls, response_id=raw_path.stem
                )
                draft = response_type.model_validate_json(response.text)
                from video_learner.notes.composition import validate_draft
                from video_learner.notes.rendering import normalize_headings

                blocks = (
                    draft.blocks
                    if isinstance(draft, Draft)
                    else [block for finding in draft.findings for block in finding.blocks]
                )
                for block in blocks:
                    block.body = normalize_headings(block.body)

                try:
                    if isinstance(draft, ReviewResult):
                        from video_learner.notes.reviewing import reviewed_chapter

                        reviewed_chapter(draft, packet)
                    else:
                        validate_draft(draft, packet)
                except TaskError as exc:
                    self.events.emit(
                        "model_validation",
                        "failed",
                        call=self.calls,
                        seconds=time.monotonic() - started,
                    )
                    if attempt == self.config.max_retries:
                        raise
                    if self._request_prefix:
                        # 历史图文可能干扰当前引用；修复轮只给目标章，避免反复使用旧证据。
                        self._request_prefix = []
                        self._history = []
                        self.events.emit("model_context", "reset", reason="validation")
                    allowed = [s["id"] for s in packet["transcript"]] + [
                        f["id"] for f in packet["frames"]
                    ]
                    repair = (
                        f"上次响应未通过语义校验：{exc}。请重新输出完整 JSON。"
                        f"本次唯一可引用的证据 ID：{json.dumps(allowed)}"
                    )
                    if isinstance(draft, ReviewResult):
                        repair += "\n待修复的完整复审结果：" + draft.model_dump_json()
                    self.events.emit(
                        "model_call",
                        "retrying",
                        attempt=attempt + 1,
                        max_retries=self.config.max_retries,
                    )
                    continue
                self.events.emit("model_call", "completed", call=self.calls)
                if packet.get("operation") == "convert":
                    if self.config.composition_context == "history":
                        # 保留实际提交的修复提示和原始成功回答，重新序列化草稿会破坏前缀。
                        self._history = self._last_input + [
                            {"role": "assistant", "content": response.text}
                        ]
                return draft
            except ValidationError as exc:
                self.events.emit(
                    "model_call",
                    "failed",
                    call=self.calls,
                    error_type=type(exc).__name__,
                    seconds=time.monotonic() - started,
                    validation_types=sorted({error["type"] for error in exc.errors()}),
                )
                problems = [{"loc": e["loc"], "type": e["type"]} for e in exc.errors()]
                repair = (
                    "上次响应未通过 JSON 结构校验："
                    + json.dumps(problems, ensure_ascii=False)
                    + "。必须返回完整对象，严格符合当前 JSON schema。"
                )
            except ProviderError as exc:
                self.events.emit(
                    "model_call",
                    "failed",
                    call=self.calls,
                    error_type=exc.error_type,
                    **({"status_code": exc.status_code} if exc.status_code is not None else {}),
                    seconds=time.monotonic() - started,
                )
                if exc.kind == "status" and exc.status_code not in (408, 409, 429):
                    if exc.status_code < 500:
                        raise InputError(
                            f"模型服务拒绝请求 (HTTP {exc.status_code})，请检查配置与权限"
                        ) from None
                if exc.kind == "service":
                    raise TaskError("模型流式服务报告错误，本次请求未完成且不自动重试") from None
            if attempt == self.config.max_retries:
                raise TaskError("模型请求失败，已达到重试上限；超时请求仍可能产生用量")
            self.events.emit(
                "model_call", "retrying", attempt=attempt + 1, max_retries=self.config.max_retries
            )
            time.sleep(min(2**attempt, 4))


def create_generator(config: Config, events: Events, *, response_directory: Path) -> NoteGenerator:
    """用所选供应商建立讲义生成会话，业务策略对两家供应商一致。"""
    return NoteGenerator(
        config, events, create_provider(config, events), response_directory=response_directory
    )
