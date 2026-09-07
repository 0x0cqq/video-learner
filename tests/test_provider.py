from types import SimpleNamespace

import pytest

from video_learner.config import Config
from video_learner.core import TaskError
from video_learner.provider import DeepSeekProvider
from video_learner.storage import Events


def packet():
    return {
        "start_us": 0,
        "end_us": 1_000_000,
        "transcript": [{"id": "tr-1", "start_us": 0, "end_us": 1_000_000, "text": "字幕"}],
        "frames": [],
        "allow_ai_additions": False,
    }


VALID = (
    '{"title":"章节","blocks":[{"kind":"text","body":"忠实整理",'
    '"category":"original","evidence_ids":["tr-1"],"frame_id":null}],"review":[]}'
)


def response(text=VALID, status="completed"):
    return SimpleNamespace(
        status=status, output_text=text, usage=SimpleNamespace(input_tokens=100, output_tokens=50)
    )


class Client:
    def __init__(self, responses):
        self.responses = self
        self.values = iter(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return next(self.values)


def test_only_deepseek_endpoint_even_with_openai_environment(tmp_path, monkeypatch):
    """设置干扰性的 OpenAI 地址，确认客户端仍固定连接 DeepSeek，且日志不包含测试密钥。"""
    import openai

    arguments = {}
    client = Client([response()])

    def constructor(**kwargs):
        arguments.update(kwargs)
        return client

    monkeypatch.setattr(openai, "OpenAI", constructor)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "local-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    provider = DeepSeekProvider(Config(), Events(tmp_path))
    provider.compose(packet(), [])
    assert arguments["base_url"] == "https://api.deepseek.com"
    assert arguments["api_key"] == "local-test-key"
    assert arguments["max_retries"] == 0
    assert client.requests[0]["reasoning"] == {"effort": "none"}
    assert "local-test-key" not in (tmp_path / ".work/logs/events.jsonl").read_text()


def test_invalid_structure_and_unknown_evidence_have_bounded_repairs(tmp_path, monkeypatch):
    """依次返回坏结构和未知引用，确认修复反馈生效且所有请求共用调用上限。"""
    monkeypatch.setattr("video_learner.provider.time.sleep", lambda _: None)
    client = Client([response("{}"), response(VALID.replace("tr-1", "unknown")), response()])
    provider = DeepSeekProvider(Config(max_retries=2), Events(tmp_path), client)
    assert provider.compose(packet(), []).blocks[0].evidence_ids == ["tr-1"]
    assert len(client.requests) == 3
    assert "语义校验" in client.requests[-1]["input"][0]["content"][-1]["text"]
    provider.config.max_calls = 3
    with pytest.raises(TaskError, match="上限"):
        provider.compose(packet(), [])
    assert len(client.requests) == 3


def test_incomplete_output_is_not_repeated_with_same_limit(tmp_path):
    """输出截断须立即失败，不能在相同输出上限下重复消耗请求。"""
    client = Client([response(status="incomplete")])
    provider = DeepSeekProvider(Config(), Events(tmp_path), client)
    with pytest.raises(TaskError, match="未完成"):
        provider.compose(packet(), [])
    assert len(client.requests) == 1
