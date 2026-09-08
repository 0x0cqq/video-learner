import json

import pytest
from test_conversion import DeterministicProvider
from typer.testing import CliRunner

from video_learner.common.config import Config, ModelPrice
from video_learner.common.core import TaskError
from video_learner.common.usage import request_cost, summarize_usage
from video_learner.workflows.conversion import convert


def test_retry_usage_and_unknown_request_are_not_free():
    """修复失败的响应仍计费，断线未知用量不当作零；ASR 秒数与图文 token 分别计价。"""
    config = Config(
        provider="qwen",
        prices={
            "qwen:qwen3.8-flash": ModelPrice(input_per_million=2, output_per_million=8),
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
    price = Config().prices["deepseek:deepseek-v4-flash-vision-exp"]
    usage = {"input_tokens": 1000, "output_tokens": 100, "cached_input_tokens": 200}
    for utc, expected in [
        ("2026-09-09T01:00:00+00:00", 0.00332),
        ("2026-09-09T04:00:00+00:00", 0.00166),
        ("2026-09-12T01:00:00+00:00", 0.00166),
    ]:
        assert request_cost({**usage, "started_at": utc}, price) == pytest.approx(expected)


def test_conversion_cli_prints_meter_and_saves_report(video, tmp_path, monkeypatch):
    """从 CLI 运行真实媒体流程，注入带用量的供应商，验证终端估价与独立 JSON。"""
    from video_learner.cli import app

    class Provider(DeterministicProvider):
        def __init__(self, events):
            self.events = events

        def compose(self, packet, images):
            """模拟服务返回已计费响应，转换仍使用真实媒体与证据校验。"""
            self.events.emit("model_call", "running", call=1, model="qwen3.8-flash")
            self.events.emit(
                "model_usage",
                "received",
                call=1,
                model="qwen3.8-flash",
                input_tokens=1000,
                output_tokens=100,
            )
            return super().compose(packet, images)

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setattr(
        "video_learner.workflows.conversion.create_provider",
        lambda config, events: Provider(events),
    )
    subtitle = video.with_suffix(".srt")
    subtitle.write_text("1\n00:00:00,000 --> 00:00:04,000\n测试\n", encoding="utf-8")
    config = tmp_path / "pricing.toml"
    config.write_text(
        'provider = "qwen"\n[prices."qwen:qwen3.8-flash"]\n'
        'currency = "CNY"\ninput_per_million = 2\noutput_per_million = 8\n',
        encoding="utf-8",
    )
    output = tmp_path / "converted"
    result = CliRunner().invoke(
        app,
        [
            "convert",
            str(video),
            "--output",
            str(output),
            "--subtitle",
            str(subtitle),
            "--config",
            str(config),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Token meter" in result.output and "0.002800 CNY" in result.output
    report = json.loads((output / "usage.json").read_text(encoding="utf-8"))
    assert report["estimate_complete"]
    assert report["input_tokens"] == 1000
    assert "test-key" not in json.dumps(report)


def test_failed_conversion_keeps_known_usage(video, tmp_path, monkeypatch):
    """单章失败也输出已经发生的用量，避免只统计成功章节而低估费用。"""

    class Provider:
        def __init__(self, events):
            self.events = events

        def compose(self, packet, images):
            """收到带用量的响应后触发校验失败，模拟已计费但未生成讲义。"""
            self.events.emit("model_call", "running", call=1)
            self.events.emit("model_usage", "received", call=1, input_tokens=50, output_tokens=20)
            raise TaskError("故障模拟")

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(
        "video_learner.workflows.conversion.create_provider",
        lambda config, events: Provider(events),
    )
    subtitle = video.with_suffix(".srt")
    subtitle.write_text("1\n00:00:00,000 --> 00:00:04,000\n测试\n", encoding="utf-8")
    output = tmp_path / "failed"
    with pytest.raises(TaskError):
        convert(video, output, Config(prices={}), subtitle=subtitle)
    report = json.loads((output / "usage.json").read_text(encoding="utf-8"))
    assert report["input_tokens"] == 50 and report["output_tokens"] == 20
    assert not report["estimate_complete"]
    assert report["estimated_known_cost"] == {}
