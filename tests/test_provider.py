from types import SimpleNamespace

import pytest

from video_learner.common.config import Config
from video_learner.common.core import TaskError
from video_learner.common.storage import Events
from video_learner.providers.base import DeepSeekProvider


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
    monkeypatch.setattr("video_learner.providers.base.time.sleep", lambda _: None)
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


def test_history_preserves_exact_successful_prefix_and_rejects_old_evidence(tmp_path):
    """追加实际成功输入/输出，历史有证据也不能绕过当前章的引用边界。"""
    client = Client([response(), response(), response()])
    provider = DeepSeekProvider(Config(max_retries=0), Events(tmp_path), client)
    first = {**packet(), "operation": "convert"}
    provider.compose(first, [])
    provider.compose(first, [])
    assert client.requests[1]["input"][:1] == client.requests[0]["input"]
    assert client.requests[1]["input"][1] == {"role": "assistant", "content": VALID}
    changed = {**first, "transcript": [{**first["transcript"][0], "id": "tr-2"}]}
    with pytest.raises(TaskError, match="以外"):
        provider.compose(changed, [])
    assert len(provider._history) == 4


def test_context_rollover_and_revision_do_not_reuse_unrelated_history(tmp_path):
    """历史超预算时整组重置；修订为独立请求，关闭历史时保持逐章行为。"""
    client = Client([response(), response(), response(), response()])
    provider = DeepSeekProvider(Config(context_token_budget=24000), Events(tmp_path), client)
    value = {**packet(), "operation": "convert", "instruction": "x" * 1500}
    provider.compose(value, [])
    provider._history[1]["content"] = "x" * 24000
    provider.compose(value, [])
    assert len(client.requests[-1]["input"]) == 1
    provider.compose({**value, "operation": "revise"}, [])
    assert len(client.requests[-1]["input"]) == 1
    provider.config.deepseek_context = "chapter"
    provider.compose(value, [])
    assert len(client.requests[-1]["input"]) == 1


def test_semantic_repair_isolates_current_chapter_from_history(tmp_path):
    """长历史导致错引时，修复轮去掉历史并列出当前允许 ID，仍计入原有调用预算。"""
    second = VALID.replace("tr-1", "tr-2")
    client = Client([response(), response(), response(second)])
    provider = DeepSeekProvider(Config(max_retries=1), Events(tmp_path), client)
    value = {**packet(), "operation": "convert"}
    provider.compose(value, [])
    provider.compose({**value, "transcript": [{**value["transcript"][0], "id": "tr-2"}]}, [])
    assert len(client.requests[1]["input"]) == 3
    assert len(client.requests[2]["input"]) == 1
    assert '"tr-2"' in client.requests[2]["input"][0]["content"][-1]["text"]
