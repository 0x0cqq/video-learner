"""多模态供应商窄接口；没有可执行工具或任意路径访问。"""

import base64
import copy
import json
import os
import time
import uuid
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from video_learner.common.config import Config, ReviewStep
from video_learner.common.core import InputError, TaskError
from video_learner.common.schemas import Draft, ReviewResult
from video_learner.common.storage import Events, atomic_bytes
from video_learner.notes.prompts import PROMPT_VERSION, SYSTEM_PROMPT
from video_learner.notes.review_prompts import review_prompt


class Provider(Protocol):
    # images 由应用层登记；返回草稿只携带证据 ID，不能决定本地路径或渲染时间戳。
    def compose(self, packet: dict, images: list[tuple[str, Path]]) -> Draft: ...

    def review(self, packet: dict, images: list[tuple[str, Path]]) -> ReviewResult: ...


def validate_provider_config(config: Config) -> None:
    """在付费请求前检查模型名和凭据是否可用，不发起远端权限或模型能力探测。"""
    if not config.model.strip():
        raise InputError("请用 TOML 的 model 或 --model 指定支持图像与结构化输出的模型")
    if not credential(config):
        raise InputError(f"未设置凭据环境变量 {config.api_key_env}；请勿将密钥写入配置或文档")


def credential(config: Config) -> str:
    """优先读取显式密钥文件，否则读取配置的环境变量；文件须只含一个非空密钥。

    返回值仅供客户端初始化，调用方不得将其写入配置、日志或异常提示。
    """
    if config.secret_file:
        try:
            path = Path(config.secret_file)
            if not path.is_file() or path.stat().st_size > 4096:
                raise OSError
            value = path.read_text(encoding="utf-8-sig").strip()
            if not value or any(c.isspace() for c in value):
                raise OSError
            return value
        except (OSError, UnicodeError):
            raise InputError("凭据文件不可读或格式无效；文件应仅包含 API Key") from None
    return os.environ.get(config.api_key_env, "")


def strict_schema(
    response_type: type[Draft] | type[ReviewResult] = Draft, step: ReviewStep | None = None
) -> dict:
    """派生响应 schema，约束块类型、frame_id 及本轮复审允许的操作。

    必填字段与额外字段由数据模型定义；复审 schema 提前排除越权输出，本地仍校验目标与证据。
    """
    schema = response_type.model_json_schema()
    block_schema = schema["$defs"]["DraftBlock"]
    alternatives = []
    kinds = (
        ("text", "figure")
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


class DeepSeekProvider:
    def __init__(self, config: Config, events: Events, client=None):
        """建立固定 DeepSeek 端点的客户端，禁用 SDK 隐式重试；client 可注入离线替身。"""
        self.config, self.events, self.calls = config, events, 0
        self._history: list[dict] = []
        self._request_prefix: list[dict] = []
        self._last_input: list[dict] = []
        if client is None:
            validate_provider_config(config)
            from openai import OpenAI

            client = OpenAI(
                api_key=credential(config),
                base_url="https://api.deepseek.com",
                max_retries=0,
                timeout=config.request_timeout_seconds,
            )
        self.client = client

    def _request(self, content: list[dict], repair: str | None):
        """将共用证据内容映射到 DeepSeek Responses 请求；repair 为本轮校验修复提示。"""
        schema = self._schema
        self._last_input = self._request_prefix + [
            {
                "role": "user",
                "content": content + ([{"type": "input_text", "text": repair}] if repair else []),
            }
        ]
        return self.client.responses.create(
            model=self.config.model,
            instructions=self._system_prompt + "\nJSON schema:\n" + json.dumps(schema),
            input=self._last_input,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "chapter",
                    "schema": schema,
                }
            },
            reasoning={"effort": self.config.reasoning_effort},
            max_output_tokens=self.config.max_output_tokens,
            store=False,
        )

    def _prepare_history(self, content: list[dict], packet: dict) -> None:
        """完整复用已成功请求前缀；到预算边界整组重置，由包内上章摘要继续衔接。

        文本以 UTF-8 字节数作保守 token 估计，DeepSeek 每图按官方上限 1024 token；
        请求体单独按 JSON 字节限制。预留 schema、结构、修复提示和输出空间。
        """
        enabled = self.config.deepseek_context == "history" and packet.get("operation") == "convert"
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
                        1024 if p["type"] == "input_image" else len(p["text"].encode("utf-8"))
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
            estimated_tokens_upper_bound=tokens,
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
        单次请求由适配器实现，Qwen 复用这里的预算、诊断及校验流程。
        """
        from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError

        step = (
            ReviewStep.model_validate(packet["review_step"])
            if response_type is ReviewResult
            else None
        )
        self._schema = strict_schema(response_type, step)
        self._system_prompt = review_prompt(step) if step is not None else SYSTEM_PROMPT
        payload_started = time.monotonic()
        text = json.dumps(packet, ensure_ascii=False)
        if len(text.encode("utf-8")) > 250_000 or len(images) > self.config.max_images_per_chapter:
            raise TaskError("单次证据包超过大小限制，请缩短章节或减少采样")
        content = [{"type": "input_text", "text": text}]
        total_image_bytes = 0
        for identity, path in images:
            total_image_bytes += path.stat().st_size
            if total_image_bytes > 25_000_000:
                raise TaskError("单次图像证据超过 25 MB，请减少候选图数量")
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.extend(
                [
                    {
                        "type": "input_text",
                        "text": "图像证据："
                        + json.dumps(
                            next(item for item in packet["frames"] if item["id"] == identity),
                            ensure_ascii=False,
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{encoded}",
                        "detail": "high",
                    },
                ]
            )
        self.events.emit(
            "model_payload",
            "prepared",
            seconds=time.monotonic() - payload_started,
            text_bytes=len(text.encode("utf-8")),
            image_bytes=total_image_bytes,
            image_count=len(images),
        )
        repair = None
        if self.config.provider == "deepseek":
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
                response = self._request(content, repair)
                usage = response.usage
                self.events.emit(
                    "model_usage",
                    "received",
                    call=self.calls,
                    model=self.config.model,
                    prompt_version=PROMPT_VERSION,
                    seconds=time.monotonic() - started,
                    input_tokens=getattr(usage, "input_tokens", None),
                    output_tokens=getattr(usage, "output_tokens", None),
                    cached_input_tokens=getattr(
                        getattr(usage, "input_tokens_details", None), "cached_tokens", None
                    ),
                )
                if response.status != "completed":
                    self.events.emit("model_call", "incomplete", call=self.calls)
                    raise TaskError("模型输出未完成，可能达到输出限制")
                raw_path = (
                    self.events.path.parent.parent / "model-responses" / f"{uuid.uuid4().hex}.json"
                )
                # 完整响应即使校验失败也保留供诊断；未完成响应不会走到这里。
                atomic_bytes(raw_path, response.output_text.encode("utf-8"))
                self.events.emit(
                    "model_response", "saved", call=self.calls, response_id=raw_path.stem
                )
                draft = response_type.model_validate_json(response.output_text)
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
                    if self.config.provider == "deepseek" and self._request_prefix:
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
                if self.config.provider == "deepseek" and packet.get("operation") == "convert":
                    if self.config.deepseek_context == "history":
                        # 保留实际提交的修复提示和原始成功回答，重新序列化草稿会破坏前缀。
                        self._history = self._last_input + [
                            {"role": "assistant", "content": response.output_text}
                        ]
                return draft
            except APIStatusError as exc:
                self.events.emit(
                    "model_call",
                    "failed",
                    call=self.calls,
                    error_type=type(exc).__name__,
                    status_code=exc.status_code,
                    seconds=time.monotonic() - started,
                )
                if exc.status_code not in (408, 409, 429) and exc.status_code < 500:
                    raise InputError(
                        f"模型服务拒绝请求 (HTTP {exc.status_code})，请检查配置与权限"
                    ) from None
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
            except (APIConnectionError, APITimeoutError) as exc:
                self.events.emit(
                    "model_call",
                    "failed",
                    call=self.calls,
                    error_type=type(exc).__name__,
                    seconds=time.monotonic() - started,
                )
            except APIError as exc:
                # 流内服务错误可能没有 HTTP 状态；记录类型和耗时，不能回显原始消息或盲目重试。
                self.events.emit(
                    "model_call",
                    "failed",
                    call=self.calls,
                    error_type=type(exc).__name__,
                    seconds=time.monotonic() - started,
                )
                raise TaskError("模型流式服务报告错误，本次请求未完成且不自动重试") from None
            if attempt == self.config.max_retries:
                raise TaskError("模型请求失败，已达到重试上限；超时请求仍可能产生用量")
            self.events.emit(
                "model_call", "retrying", attempt=attempt + 1, max_retries=self.config.max_retries
            )
            time.sleep(min(2**attempt, 4))


def create_provider(config: Config, events: Events) -> Provider:
    """按已校验配置选择图文适配器，不探测或自动回退到其他供应商。"""
    if config.provider == "qwen":
        from video_learner.providers.qwen import QwenProvider

        return QwenProvider(config, events)
    return DeepSeekProvider(config, events)
