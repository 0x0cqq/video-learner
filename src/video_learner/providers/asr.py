"""阿里云 Qwen 语音识别适配器；输入 WAV，返回完整窗口文本。"""

import base64
import time
from threading import Event, Lock
from typing import Protocol

from video_learner.common.config import Config
from video_learner.common.core import InputError, TaskError
from video_learner.common.storage import Events
from video_learner.providers.base import credential


class Recognizer(Protocol):
    def recognize(self, wav: bytes, cancelled: Event | None = None) -> str: ...

    def close(self) -> None: ...


def qwen_credential(config: Config) -> str:
    """复用密钥读取规则但替换为 ASR 专用配置，避免误用图文供应商凭据。"""
    return credential(
        config.model_copy(
            update={"api_key_env": config.asr_api_key_env, "secret_file": config.asr_secret_file}
        )
    )


def validate_qwen_config(config: Config) -> None:
    """预检同步识别模型和凭据，不解码媒体或规划切片。"""
    if (
        not config.asr_qwen_model.startswith("qwen3-asr-flash")
        or "filetrans" in config.asr_qwen_model
    ):
        raise InputError("Qwen 兼容接口须使用 qwen3-asr-flash 或其快照，不能使用 filetrans")
    if not qwen_credential(config):
        raise InputError("Qwen ASR 缺少 DASHSCOPE_API_KEY 或 --asr-secret 凭据")


class QwenASR:
    def __init__(self, config: Config, events: Events, client=None):
        """建立固定 DashScope 端点的独立 ASR 客户端；由本层统计实际请求及重试。"""
        self.config, self.events, self.calls = config, events, 0
        self._call_lock = Lock()
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                api_key=qwen_credential(config),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                max_retries=0,
                timeout=config.request_timeout_seconds,
            )
        self.client = client

    def close(self) -> None:
        self.client.close()

    def recognize(self, wav: bytes, cancelled: Event | None = None) -> str:
        """发送一段 WAV 并返回完整转写文本；临时服务错误有界重试，不完整输出直接失败。

        接口不提供句级时间，因此这里只返回文本，时间映射由音频窗口编排层保存。
        调用编号和总预算在锁内分配；取消会唤醒退避，但不强制中断正在进行的 HTTP 请求。
        """
        from openai import APIConnectionError, APIStatusError, APITimeoutError

        cancelled = cancelled or Event()
        data = "data:audio/wav;base64," + base64.b64encode(wav).decode("ascii")
        options = {"enable_itn": False}
        if self.config.asr_language:
            options["language"] = self.config.asr_language
        for attempt in range(self.config.max_retries + 1):
            with self._call_lock:
                if cancelled.is_set():
                    raise TaskError("Qwen ASR 已取消")
                if self.calls >= self.config.asr_max_calls:
                    raise TaskError("Qwen ASR 已达到调用上限；未完成转写不能发布成功版本")
                self.calls += 1
                call = self.calls
                started = time.monotonic()
                self.events.emit("asr_model_call", "running", call=call, attempt=attempt + 1)
            try:
                response = self.client.chat.completions.create(
                    model=self.config.asr_qwen_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [{"type": "input_audio", "input_audio": {"data": data}}],
                        }
                    ],
                    stream=False,
                    extra_body={"asr_options": options},
                )
                usage = response.usage
                self.events.emit(
                    "asr_model_usage",
                    "received",
                    call=call,
                    model=self.config.asr_qwen_model,
                    seconds=time.monotonic() - started,
                    input_tokens=getattr(usage, "prompt_tokens", None),
                    output_tokens=getattr(usage, "completion_tokens", None),
                    audio_seconds=getattr(usage, "seconds", None),
                )
                if not response.choices or response.choices[0].finish_reason != "stop":
                    raise TaskError("Qwen ASR 输出未完整结束")
                text = response.choices[0].message.content
                if not isinstance(text, str) or len(text) > 20000:
                    raise TaskError("Qwen ASR 返回了无效转写文本")
                self.events.emit("asr_model_call", "completed", call=call)
                return text.strip()
            except APIStatusError as exc:
                self.events.emit(
                    "asr_model_call",
                    "failed",
                    call=call,
                    status_code=exc.status_code,
                    seconds=time.monotonic() - started,
                )
                if exc.status_code not in (408, 409, 429) and exc.status_code < 500:
                    raise InputError(f"Qwen ASR 拒绝请求 (HTTP {exc.status_code})") from None
            except (APIConnectionError, APITimeoutError) as exc:
                self.events.emit(
                    "asr_model_call",
                    "failed",
                    call=call,
                    error_type=type(exc).__name__,
                    seconds=time.monotonic() - started,
                )
            if attempt == self.config.max_retries:
                raise TaskError("Qwen ASR 请求失败，已达到重试上限；超时请求仍可能计费")
            self.events.emit(
                "asr_model_call",
                "retrying",
                attempt=attempt + 1,
                max_retries=self.config.max_retries,
            )
            if cancelled.wait(min(2**attempt, 4)):
                raise TaskError("Qwen ASR 已取消")
