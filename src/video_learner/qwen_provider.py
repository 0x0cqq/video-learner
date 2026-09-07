"""Qwen 3.8 Flash 图文流式适配；思考增量不写入讲义或诊断正文。"""

import json
import time
from types import SimpleNamespace

from .config import Config
from .core import TaskError
from .provider import (
    SYSTEM_PROMPT,
    DeepSeekProvider,
    credential,
    strict_schema,
    validate_provider_config,
)
from .storage import Events


class QwenProvider(DeepSeekProvider):
    def __init__(self, config: Config, events: Events, client=None):
        """建立固定 DashScope 端点的图文客户端，沿用共用请求预算并支持离线客户端注入。"""
        self.config, self.events, self.calls = config, events, 0
        if client is None:
            from openai import OpenAI

            validate_provider_config(config)
            client = OpenAI(
                api_key=credential(config),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                max_retries=0,
                timeout=config.request_timeout_seconds,
            )
        self.client = client

    def _request(self, content: list[dict], repair: str | None):
        """将共用图像证据转换成 Chat Completions 流，归一化为共用校验层的响应。

        思考增量只触发进度，不保存文本；拼接最终正文并处理仅含 usage 的尾块。
        缺少正常停止标记视为未完成，超时或异常关闭流，局部正文不会带入下一次请求。
        """
        parts = []
        for item in content:
            if item["type"] == "input_text":
                parts.append({"type": "text", "text": item["text"]})
            else:
                parts.append({"type": "image_url", "image_url": {"url": item["image_url"]}})
        if repair:
            parts.append({"type": "text", "text": repair})
        options = {"enable_thinking": self.config.qwen_enable_thinking}
        if self.config.qwen_enable_thinking:
            options["thinking_budget"] = self.config.qwen_thinking_budget
        started = time.monotonic()
        stream = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT + "\nJSON schema:\n" + json.dumps(strict_schema()),
                },
                {"role": "user", "content": parts},
            ],
            stream=True,
            stream_options={"include_usage": True},
            extra_body=options,
            max_completion_tokens=self.config.max_output_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "chapter", "schema": strict_schema(), "strict": True},
            },
        )
        answer, size, finish, usage = [], 0, None, None
        phases = set()
        try:
            for chunk in stream:
                if time.monotonic() - started > self.config.request_timeout_seconds:
                    raise TaskError("Qwen 流式响应超过时间上限")
                if getattr(chunk, "usage", None) is not None:
                    usage = chunk.usage
                if not chunk.choices:
                    # 服务可在正文结束后另发 usage 尾块，它没有可读取的 delta。
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                for phase, value in (
                    ("thinking", getattr(delta, "reasoning_content", None)),
                    ("answering", getattr(delta, "content", None)),
                ):
                    if value and phase not in phases:
                        self.events.emit("model_stream", phase, call=self.calls)
                        phases.add(phase)
                if getattr(delta, "content", None):
                    size += len(delta.content)
                    if size > 100000:
                        raise TaskError("Qwen 正文超过大小上限")
                    answer.append(delta.content)
                if choice.finish_reason is not None:
                    finish = choice.finish_reason
        finally:
            stream.close()
        return SimpleNamespace(
            status="completed" if finish == "stop" and answer else "incomplete",
            output_text="".join(answer),
            usage=SimpleNamespace(
                input_tokens=getattr(usage, "prompt_tokens", None),
                output_tokens=getattr(usage, "completion_tokens", None),
            ),
        )
