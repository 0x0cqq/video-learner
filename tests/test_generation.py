import pytest
from model_fakes import VALID, packet, session

from video_learner.common.core import InputError, TaskError

pytestmark = pytest.mark.parametrize("supplier", ["deepseek", "qwen"])


def messages(client, index):
    """取出线上请求实际携带的消息，排除千问单列的系统消息。"""
    request = client.requests[index]
    return request["input"] if "input" in request else request["messages"][1:]


def test_invalid_structure_and_unknown_evidence_have_bounded_repairs(
    tmp_path, monkeypatch, supplier
):
    """坏结构和未知引用由生成层修复；两家请求与修复共用调用上限。"""
    monkeypatch.setattr("video_learner.notes.generation.time.sleep", lambda _: None)
    generator, client = session(
        tmp_path,
        supplier,
        ["{}", VALID.replace("tr-1", "unknown"), VALID.replace("忠实整理", ""), VALID],
        max_retries=3,
    )
    assert generator.compose(packet(), []).blocks[0].evidence_ids == ["tr-1"]
    assert len(client.requests) == 4
    assert "语义校验" in messages(client, -1)[0]["content"][-1]["text"]
    generator.config.max_calls = 4
    with pytest.raises(TaskError, match="上限"):
        generator.compose(packet(), [])
    assert len(client.requests) == 4


def test_response_storage_is_independent_of_log_location(tmp_path, supplier):
    """迁移事件日志后，最终响应仍写入显式指定的目录，两家供应商行为一致。"""
    generator, _ = session(tmp_path, supplier, [VALID])
    log_directory = tmp_path / "external-logs"
    log_directory.mkdir()
    generator.events.path = log_directory / "events.jsonl"
    generator.compose(packet(), [])
    assert [
        path.read_text(encoding="utf-8") for path in generator.response_directory.glob("*.json")
    ] == [VALID]
    assert (log_directory / "events.jsonl").is_file()
    assert not (tmp_path / "model-responses").exists()


def test_context_rollover_and_revision_do_not_reuse_unrelated_history(tmp_path, supplier):
    """同一预算触发历史清空；修订及 chapter 模式对两家都使用独立请求。"""
    generator, client = session(tmp_path, supplier, [VALID] * 4, context_token_budget=24000)
    value = {**packet(), "operation": "convert", "instruction": "x" * 1500}
    generator.compose(value, [])
    generator._history[1]["content"] = "x" * 24000
    generator.compose(value, [])
    assert len(messages(client, -1)) == 1
    generator.compose({**value, "operation": "revise"}, [])
    assert len(messages(client, -1)) == 1
    generator.config.composition_context = "chapter"
    generator.compose(value, [])
    assert len(messages(client, -1)) == 1


@pytest.mark.parametrize("status", [401, 429, 503])
def test_provider_errors_keep_common_retry_policy_and_hide_service_body(
    tmp_path, supplier, status, monkeypatch
):
    """受控错误保留状态供生成层决策；认证错误立即停止，临时失败重试且不泄漏服务正文。"""
    import httpx2
    from openai import APIStatusError

    monkeypatch.setattr("video_learner.notes.generation.time.sleep", lambda _: None)
    error = APIStatusError(
        "private-provider-error",
        response=httpx2.Response(status, request=httpx2.Request("POST", "https://example.invalid")),
        body={"message": "private-provider-error"},
    )
    generator, client = session(tmp_path, supplier, [error, VALID], max_retries=1)
    if status == 401:
        with pytest.raises(InputError, match="HTTP 401"):
            generator.compose(packet(), [])
        assert len(client.requests) == 1
    else:
        assert generator.compose(packet(), []).title == "章节"
        assert len(client.requests) == 2
    log = (tmp_path / ".work/logs/events.jsonl").read_text(encoding="utf-8")
    assert "private-provider-error" not in log
