"""供应商无关的单次请求契约、凭据读取与适配器选择。"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from video_learner.common.config import Config
from video_learner.common.core import InputError
from video_learner.common.storage import Events


@dataclass(frozen=True)
class ModelRequest:
    """业务层已经确定的提示词、响应结构及完整消息；适配器逐项转换。"""

    instructions: str
    schema: dict
    messages: list[dict]


@dataclass(frozen=True)
class ModelUsage:
    """服务端已返回的计量，成功与失败通道均可携带；空值表示未知。"""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None


@dataclass(frozen=True)
class ModelResponse:
    """最终正文、完成状态及服务端计量；不含内部思考文本。"""

    text: str
    completed: bool
    usage: ModelUsage = field(default_factory=ModelUsage)


class Provider(Protocol):
    def generate(self, request: ModelRequest, *, call: int) -> ModelResponse: ...


class ProviderError(Exception):
    """携带受控错误和已知计量，供业务层决定重试；不携带服务端敏感正文。"""

    def __init__(
        self,
        kind: Literal["status", "connection", "service", "incomplete", "cancelled"],
        error_type: str,
        status_code: int | None = None,
        *,
        usage: ModelUsage | None = None,
    ):
        super().__init__(error_type)
        self.kind, self.error_type, self.status_code = kind, error_type, status_code
        self.usage = usage


@contextmanager
def provider_errors() -> Iterator[None]:
    """把兼容 SDK 异常映射为单次请求错误；重试策略留给业务层。"""
    from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError

    try:
        yield
    except APIStatusError as exc:
        raise ProviderError("status", type(exc).__name__, exc.status_code) from None
    except (APIConnectionError, APITimeoutError) as exc:
        raise ProviderError("connection", type(exc).__name__) from None
    except APIError as exc:
        raise ProviderError("service", type(exc).__name__) from None


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


def create_provider(config: Config, events: Events) -> Provider:
    """只选择单次请求适配器；任务与上下文策略由讲义生成层决定。"""
    if config.provider == "qwen":
        from video_learner.providers.qwen import QwenProvider

        return QwenProvider(config, events)
    from video_learner.providers.deepseek import DeepSeekProvider

    return DeepSeekProvider(config, events)
