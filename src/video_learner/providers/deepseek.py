"""DeepSeek Responses API 的单次图文请求适配。"""

import json

from video_learner.common.config import Config
from video_learner.common.storage import Events
from video_learner.providers.base import (
    ModelRequest,
    ModelResponse,
    ModelUsage,
    credential,
    provider_errors,
    validate_provider_config,
)


class DeepSeekProvider:
    def __init__(self, config: Config, events: Events, client=None):
        """固定连接 DeepSeek 并关闭 SDK 重试；client 可注入离线替身。"""
        self.config = config
        if client is None:
            from openai import OpenAI

            validate_provider_config(config)
            client = OpenAI(
                api_key=credential(config),
                base_url="https://api.deepseek.com",
                max_retries=0,
                timeout=config.request_timeout_seconds,
            )
        self.client = client

    def generate(self, request: ModelRequest, *, call: int) -> ModelResponse:
        """逐条转换业务层已选定的消息；返回最终正文、完成状态和服务端用量。"""
        with provider_errors():
            messages = []
            for message in request.messages:
                content = message["content"]
                if isinstance(content, list):
                    content = [
                        {"type": "input_text", "text": part["text"]}
                        if part["type"] == "text"
                        else {
                            "type": "input_image",
                            "image_url": part["image_url"]["url"],
                            "detail": part["image_url"]["detail"],
                        }
                        for part in content
                    ]
                messages.append({"role": message["role"], "content": content})
            response = self.client.responses.create(
                model=self.config.model,
                instructions=request.instructions + "\nJSON schema:\n" + json.dumps(request.schema),
                input=messages,
                text={
                    "format": {"type": "json_schema", "name": "chapter", "schema": request.schema}
                },
                reasoning={"effort": self.config.reasoning_effort},
                max_output_tokens=self.config.max_output_tokens,
                store=False,
            )
            usage = response.usage
            return ModelResponse(
                text=response.output_text,
                completed=response.status == "completed",
                usage=ModelUsage(
                    input_tokens=getattr(usage, "input_tokens", None),
                    output_tokens=getattr(usage, "output_tokens", None),
                    cached_input_tokens=getattr(
                        getattr(usage, "input_tokens_details", None), "cached_tokens", None
                    ),
                ),
            )
