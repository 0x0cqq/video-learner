"""可审查的 TOML 配置；不读取或持久化密钥值。"""

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from video_learner.common.core import InputError
from video_learner.common.schemas import Record


class ModelPrice(Record):
    """用户提供的单模型单价；空值表示尚未提供，不代表免费。"""

    currency: str = Field(default="CNY", min_length=1, max_length=12)
    input_per_million: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    output_per_million: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    cached_input_per_million: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    audio_per_second: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    deepseek_off_peak_multiplier: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def billing_unit(self):
        """拒绝同时指定音频和 token 单价，避免无声忽略其中一种计费规则。"""
        if self.audio_per_second is not None and any(
            value is not None
            for value in (
                self.input_per_million,
                self.output_per_million,
                self.cached_input_per_million,
            )
        ):
            raise ValueError("音频按秒计价与 token 计价不能同时配置")
        return self


def default_prices() -> dict[str, ModelPrice]:
    """读取随包保存的公开价格快照；用户的 prices 表可整体替换它。"""
    data = tomllib.loads(Path(__file__).with_name("prices.toml").read_text(encoding="utf-8"))
    return {key: ModelPrice.model_validate(value) for key, value in data["prices"].items()}


class Config(Record):
    prices: dict[str, ModelPrice] = Field(default_factory=default_prices)
    provider: Literal["deepseek", "qwen"] = "deepseek"
    qwen_enable_thinking: bool = True
    qwen_thinking_budget: int = Field(default=1024, ge=128, le=8192)

    @model_validator(mode="before")
    @classmethod
    def provider_defaults(cls, data):
        """按供应商补齐未显式指定的模型和凭据环境名，保留调用者的明确设置。"""
        if isinstance(data, dict) and data.get("provider") == "qwen":
            data = dict(data)
            data.setdefault("model", "qwen3.8-flash")
            data.setdefault("api_key_env", "DASHSCOPE_API_KEY")
        return data

    profile: Literal["programming", "math", "mixed"] = "mixed"
    instruction: str = ""
    allow_ai_additions: bool = False
    asr_language: str | None = "zh"
    asr_qwen_model: str = "qwen3-asr-flash"
    asr_api_key_env: str = Field(default="DASHSCOPE_API_KEY", pattern=r"^[A-Za-z_][A-Za-z_0-9]*$")
    asr_secret_file: str | None = None
    asr_window_seconds: int = Field(default=30, ge=5, le=60)
    asr_max_calls: int = Field(default=500, ge=1, le=10000)
    jobs: int = Field(default=1, ge=1, le=32)
    sample_seconds: int = Field(default=2, ge=1, le=120)
    chapter_seconds: int = Field(default=180, ge=30, le=600)
    max_images_per_chapter: int = Field(default=12, ge=1, le=30)
    image_change_threshold: float = Field(default=0.035, ge=0, le=1)
    model: str = "deepseek-v4-flash-vision-exp"
    api_key_env: str = Field(default="DEEPSEEK_API_KEY", pattern=r"^[A-Za-z_][A-Za-z_0-9]*$")
    secret_file: str | None = None
    max_calls: int = Field(default=80, ge=1, le=1000)
    request_timeout_seconds: int = Field(default=90, ge=1, le=300)
    max_retries: int = Field(default=2, ge=0, le=4)
    max_output_tokens: int = Field(default=6000, ge=500, le=16000)
    reasoning_effort: Literal["none", "low", "high", "max"] = "none"


def merge_provider_settings(base: dict, updates: dict) -> dict:
    """合并非空覆盖值；切换供应商时先移除继承的模型和凭据设置。

    避免把旧供应商的密钥发往新服务，再由 Config 补齐新供应商默认值。
    """
    updates = {key: value for key, value in updates.items() if value is not None}
    result = dict(base)
    if updates.get("provider", base.get("provider", "deepseek")) != base.get(
        "provider", "deepseek"
    ):
        for key in ("model", "api_key_env", "secret_file"):
            result.pop(key, None)
    result.update(updates)
    return result


def load_config(path: Path | None = None, **overrides) -> Config:
    """按覆盖参数 > TOML > 默认值加载并校验配置，错误提示仅包含字段位置。"""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8")) if path else {}
        data = merge_provider_settings(data, overrides)
        return Config.model_validate(data)
    except (OSError, ValueError) as exc:
        # Pydantic 完整错误可能带输入值；提示只列字段位置，避免回显敏感配置。
        fields = (
            ", ".join(str(e["loc"]) for e in exc.errors())
            if isinstance(exc, ValidationError)
            else ""
        )
        raise InputError(f"配置无效，请检查 TOML 字段及范围 {fields}") from exc
