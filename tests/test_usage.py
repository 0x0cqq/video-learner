import pytest

from video_learner.common.config import Config, ModelPrice
from video_learner.common.usage import request_cost, summarize_usage


@pytest.mark.parametrize("provider", ["deepseek", "qwen"])
def test_retry_usage_and_unknown_request_are_not_free(provider):
    """修复失败的响应仍计费，断线未知用量不当作零；ASR 秒数与图文 token 分别计价。"""
    config = Config(
        provider=provider,
        prices={
            f"{provider}:{Config(provider=provider).model}": ModelPrice(
                input_per_million=2, output_per_million=8
            ),
            "qwen:qwen3-asr-flash": ModelPrice(audio_per_second=0.01),
        },
    )
    records = [
        {"stage": "model_call", "status": "running", "call": 1},
        {
            "stage": "model_usage",
            "status": "received",
            "call": 1,
            "input_tokens": 1000,
            "output_tokens": 100,
        },
        {"stage": "model_call", "status": "running", "call": 2},
        {"stage": "asr_model_call", "status": "running", "call": 1},
        {
            "stage": "asr_model_usage",
            "status": "received",
            "call": 1,
            "input_tokens": 400,
            "output_tokens": 30,
            "audio_seconds": 20,
        },
    ]
    report = summarize_usage(records, config)
    assert [(row["provider"], row["model"]) for row in report["models"]] == [
        (provider, config.model),
        ("qwen", config.asr_qwen_model),
    ]
    assert report["input_tokens"] == 1400
    assert report["output_tokens"] == 130
    assert report["estimated_known_cost"]["CNY"] == pytest.approx(0.2028)
    assert report["missing_token_usage"] == 1
    assert not report["estimate_complete"]
    assert report["models"][0]["unpriced_requests"] == 1
    report = summarize_usage([r for r in records if r["call"] != 2], config)
    assert report["estimate_complete"]


def test_cache_price_requires_usage_and_does_not_add_tokens_twice():
    """缓存是输入子项；缺失单价/用量不能估成免费，显式零价则允许。"""
    price = ModelPrice(input_per_million=2, output_per_million=8, cached_input_per_million=0.2)
    usage = {"input_tokens": 1000, "output_tokens": 100, "cached_input_tokens": 600}
    assert request_cost(usage, price) == pytest.approx(0.00172)
    assert request_cost({"input_tokens": 1000, "output_tokens": 100}, price) == pytest.approx(
        0.0028
    )
    assert request_cost(usage, ModelPrice()) is None
    assert request_cost(usage, ModelPrice(input_per_million=0, output_per_million=0)) == 0


def test_deepseek_prices_follow_beijing_request_start():
    """用 UTC 请求时间覆盖高峰、午休和周末，缓存折扣与时段折扣各只应用一次。"""
    price = Config().prices["deepseek:deepseek-flash"]
    usage = {"input_tokens": 1000, "output_tokens": 100, "cached_input_tokens": 200}
    for utc, expected in [
        ("2026-09-09T01:00:00+00:00", 0.002408),
        ("2026-09-09T04:00:00+00:00", 0.001204),
        ("2026-09-12T01:00:00+00:00", 0.001204),
    ]:
        assert request_cost({**usage, "started_at": utc}, price) == pytest.approx(expected)
