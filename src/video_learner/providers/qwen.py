"""Qwen 3.8 Flash 图文流式适配；思考增量不写入讲义或诊断正文。"""

import json
import time
from types import SimpleNamespace

from video_learner.common.config import Config
from video_learner.common.core import TaskError
from video_learner.common.storage import Events
from video_learner.providers.base import (
    SYSTEM_PROMPT,
    DeepSeekProvider,
    credential,
    strict_schema,
    validate_provider_config,
)


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
        stream_open_seconds = time.monotonic() - started
        answer, size, finish, usage = [], 0, None, None
        first_chunk = first_reasoning = first_answer = None
        chunks = reasoning_characters = 0
        exhausted = False
        phases = set()
        try:
            for chunk in stream:
                elapsed = time.monotonic() - started
                chunks += 1
                if first_chunk is None:
                    first_chunk = elapsed
                if elapsed > self.config.request_timeout_seconds:
                    raise TaskError("Qwen 流式响应超过时间上限")
                if getattr(chunk, "usage", None) is not None:
                    usage = chunk.usage
                if not chunk.choices:
                    # 服务可在正文结束后另发 usage 尾块，它没有可读取的 delta。
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if getattr(delta, "reasoning_content", None):
                    reasoning_characters += len(delta.reasoning_content)
                    if first_reasoning is None:
                        first_reasoning = elapsed
                if getattr(delta, "content", None) and first_answer is None:
                    first_answer = elapsed
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
            exhausted = True
        finally:
            stream.close()
            # 只记录客户端观察到的阶段延迟和字符数，不保存思考文本，也不推断服务端推理耗时。
            self.events.emit(
                "model_stream_timing",
                "completed" if exhausted and finish == "stop" and answer else "incomplete",
                call=self.calls,
                seconds=time.monotonic() - started,
                stream_open_seconds=stream_open_seconds,
                first_chunk_seconds=first_chunk,
                first_reasoning_seconds=first_reasoning,
                first_answer_seconds=first_answer,
                chunks=chunks,
                answer_characters=size,
                reasoning_characters=reasoning_characters,
            )
        return SimpleNamespace(
            status="completed" if finish == "stop" and answer else "incomplete",
            output_text="".join(answer),
            usage=SimpleNamespace(
                input_tokens=getattr(usage, "prompt_tokens", None),
                output_tokens=getattr(usage, "completion_tokens", None),
            ),
        )
