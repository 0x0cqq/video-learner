import json

import pytest
from test_conversion import DeterministicProvider
from test_conversion import converted as converted
from test_provider import Client, response

from video_learner.common.config import Config, ReviewStep, load_config
from video_learner.common.core import InputError, TaskError
from video_learner.common.schemas import Draft, DraftBlock, Notebook, ReviewFinding, ReviewResult
from video_learner.common.storage import Events, read_json
from video_learner.notes.composition import evidence_packet, validate_draft
from video_learner.notes.rendering import expected_spans, image_dependencies, render_notes
from video_learner.notes.reviewing import apply_review_pass, review_input, reviewed_chapter
from video_learner.providers.base import DeepSeekProvider
from video_learner.workflows.conversion import convert
from video_learner.workflows.replay import replay_conversion, replay_revision
from video_learner.workflows.reviewing import review


def prepared(root):
    """构造真实自造素材的复审包，测试不调用云端或读取私人课程。"""
    book = Notebook.model_validate(read_json(root / "notes.json"))
    current = (root / "notes.md").read_bytes()
    frozen, images = review_input(
        book, book.chapters[0], Config(sample_seconds=1), root, current, "r002"
    )
    return book, current, frozen.packet, images


class EditingProvider(DeterministicProvider):
    def review(self, packet, images):
        """只替换首个解释块，保留相邻图片，供非目标字节和版本隔离检查。"""
        result = super().review(packet, images)
        if packet.get("review_step", {}).get("kind") == "visual":
            return result
        block = packet["chapter"]["blocks"][0]
        replacement = DraftBlock(**{key: block[key] for key in DraftBlock.model_fields})
        replacement.body = "修正后的讲义说明。"
        result.findings = [
            ReviewFinding(
                target_id=block["id"],
                kind="fidelity",
                reason="对照原证据修正解释。",
                evidence_ids=block["evidence_ids"],
                action="replace",
                blocks=[replacement],
            )
        ]
        return result


def test_review_preserves_baseline_and_non_target_manual_bytes(converted):
    """手改图片注释、CRLF 和附加文字逐字保留；新版本及复审可独立导出和离线重放。"""
    root, video = converted
    path = root / "notes.md"
    before = (
        path.read_bytes()
        .replace("变化后的原视频画面。".encode(), "手改的保留图注。".encode())
        .replace(b"\n", b"\r\n")
    )
    path.write_bytes(before)
    destination = review(root, provider=EditingProvider())
    assert path.read_bytes() == before
    after = (destination / "notes.md").read_bytes()
    original = Notebook.model_validate(read_json(root / "notes.json"))
    updated = Notebook.model_validate(read_json(destination / "notes.json"))
    old, new = (
        expected_spans(before, original)["blk-001-001"],
        expected_spans(after, updated)["blk-001-001"],
    )
    assert before[: old.start] == after[: new.start]
    assert before[old.end :] == after[new.end :]
    assert updated.chapters[0].blocks[1].sync_status == "manual_unverified"
    assert updated.chapters[0].blocks[0].sync_status == "generated"
    for reference in image_dependencies(after):
        assert (destination / reference).is_file()
    video.unlink()
    assert replay_conversion(root) == original
    assert replay_revision(root, "r002") == updated
    manifest = read_json(root / ".work/manifest.json")
    assert manifest["versions"]["r002"]["base"] == "r001"


def test_default_conversion_returns_reviewed_version(video, tmp_path):
    """默认转换须走完初稿及独立复审两阶段，并同时保留两个可重放版本。"""
    subtitle = video.with_suffix(".srt")
    subtitle.write_text("1\n00:00:00,000 --> 00:00:04,000\n自造说明。\n", encoding="utf-8")
    root = tmp_path / "reviewed"
    destination = convert(video, root, Config(), subtitle=subtitle, provider=EditingProvider())
    assert destination == root / "revisions/r002"
    assert "修正后的" in (destination / "notes.md").read_text(encoding="utf-8")
    assert "修正后的" not in (root / "notes.md").read_text(encoding="utf-8")
    replay_conversion(root)
    replay_revision(root, "r002")


@pytest.mark.parametrize("failure", [TaskError("测试复审失败"), KeyboardInterrupt()])
def test_failed_review_keeps_successful_initial_version(video, tmp_path, failure):
    """复审异常或取消不能撤销已提交初稿，也不能登记半个复审版本。"""

    class Failed(DeterministicProvider):
        def review(self, packet, images):
            """第一轮成功后模拟第二轮失败，避免发布半条复审流程。"""
            if packet["review_step"]["kind"] == "visual":
                return super().review(packet, images)
            raise failure

    subtitle = video.with_suffix(".srt")
    subtitle.write_text("1\n00:00:00,000 --> 00:00:04,000\n说明\n", encoding="utf-8")
    root = tmp_path / "failed"
    with pytest.raises(type(failure)):
        convert(video, root, Config(), subtitle=subtitle, provider=Failed())
    manifest = read_json(root / ".work/manifest.json")
    assert manifest["status"] == "completed" and set(manifest["versions"]) == {"r001"}
    assert (root / "usage.json").is_file()
    replay_conversion(root)


def test_concurrent_edit_blocks_review_commit(converted):
    """工具持锁也须发现编辑器在模型请求期间的修改。"""
    root, _ = converted

    class Concurrent(EditingProvider):
        def review(self, packet, images):
            """模拟用户在另一个编辑器追加笔记。"""
            path = root / "notes.md"
            path.write_bytes(path.read_bytes() + b"\nmanual\n")
            return super().review(packet, images)

    with pytest.raises(InputError, match="并发编辑"):
        review(root, provider=Concurrent())
    assert set(read_json(root / ".work/manifest.json")["versions"]) == {"r001"}
    assert (root / "notes.md").read_bytes().endswith(b"manual\n")


def test_review_decisions_and_patches_must_agree(converted):
    """遗漏候选图、凭空选图、越界证据、重复目标均不能变成成功讲义。"""
    root, _ = converted
    book, current, packet, images = prepared(root)
    result = EditingProvider().review(packet, images)
    changed = result.model_copy(deep=True)
    changed.frames.pop()
    with pytest.raises(TaskError, match="全部候选图"):
        reviewed_chapter(changed, packet)
    changed = result.model_copy(deep=True)
    changed.findings[0].blocks = []
    with pytest.raises(TaskError, match="保留的块"):
        reviewed_chapter(changed, packet)
    changed = result.model_copy(deep=True)
    changed.findings[0].evidence_ids = ["tr-outside"]
    with pytest.raises(TaskError, match="越界"):
        reviewed_chapter(changed, packet)
    changed = result.model_copy(deep=True)
    changed.findings.append(changed.findings[0])
    with pytest.raises(TaskError, match="多次"):
        reviewed_chapter(changed, packet)
    # 只提交逐图决定，应用层自动换图并插在解释后，无需第二份插图补丁。
    result.frames[0].decision = "omit"
    result.frames[-1].decision = "use"
    identity = result.frames[-1].frame_id
    updated, generated = apply_review_pass(book, packet, result, current)
    assert len(image_dependencies(generated)) == 1
    assert updated.chapters[0].blocks[-1].frame_id == identity
    assert book.chapters[0].blocks[-1].frame_id != identity


def test_body_ids_and_orphan_headings_are_validated_without_word_filter(converted):
    """拒绝元数据和空标题，代码字面量和有图无图注的小节仍合法。"""
    root, _ = converted
    book, _, packet, _ = prepared(root)
    draft = Draft(
        title="标题",
        blocks=[
            DraftBlock(**b.model_dump(include=set(DraftBlock.model_fields)))
            for b in book.chapters[0].blocks
        ],
        review=[],
    )
    draft.blocks[0].body = f"参见 `{packet['frames'][0]['id']}`。"
    with pytest.raises(TaskError, match="内部证据"):
        validate_draft(draft, packet)
    draft.blocks[0].body = "```text\n" + packet["frames"][0]["id"] + "\n```"
    validate_draft(draft, packet)
    draft.blocks[0].body = "### 完整画面"
    validate_draft(draft, packet)
    draft.blocks = draft.blocks[:1]
    with pytest.raises(TaskError, match="只有标题"):
        validate_draft(draft, packet)


def test_review_preserves_manual_text_and_visual_only_chapter(converted):
    """手改正文以 Markdown 为准；纯图章节仍可复审，不强行捏造语音或文字。"""
    root, _ = converted
    book, current, packet, images = prepared(root)
    packet["chapter"]["blocks"][0].update(
        body=packet["frames"][0]["id"], sync_status="manual_unverified"
    )
    book.chapters[0].blocks[0].body = packet["frames"][0]["id"]
    book.chapters[0].blocks[0].sync_status = "manual_unverified"
    result = DeterministicProvider().review(packet, images)
    result.frames[0].related_block_id = packet["chapter"]["blocks"][1]["id"]
    assert apply_review_pass(book, packet, result, current)[1] == current
    packet["chapter"]["blocks"] = packet["chapter"]["blocks"][1:]
    result = DeterministicProvider().review(packet, images)
    assert len(reviewed_chapter(result, packet).blocks) == 1


def test_picture_timing_is_real_window_and_not_sentence_alignment(converted):
    """按原微秒时间映射到真实切片；前后文保留方向但不加入可引用转写。"""
    root, _ = converted
    book, _, _, _ = prepared(root)
    segment = book.transcript[0]
    segment.end_us = 1_000_000
    book.transcript.extend(
        [
            segment.model_copy(
                update={"id": "tr-middle", "start_us": 1_000_000, "end_us": 3_000_000}
            ),
            segment.model_copy(
                update={"id": "tr-after", "start_us": 3_000_000, "end_us": 4_000_000}
            ),
        ]
    )
    chapter = book.chapters[0].model_copy(update={"start_us": 1_000_000, "end_us": 3_000_000})
    packet, _ = evidence_packet(book, chapter, Config(), root)
    assert [s["id"] for s in packet["transcript"]] == ["tr-middle"]
    assert packet["boundary_context_not_citable"]["before"][0]["id"] == segment.id
    assert packet["boundary_context_not_citable"]["after"][0]["id"] == "tr-after"
    assert all(f["speech_window_ids"] == ["tr-middle"] for f in packet["frames"])


@pytest.mark.parametrize("supplier", ["deepseek", "qwen"])
def test_independent_review_schema_and_repairs(converted, supplier, monkeypatch):
    """两家供应商共用复审约束；修复不继承生成历史且计入同一调用预算。"""
    monkeypatch.setattr("video_learner.providers.base.time.sleep", lambda _: None)
    root, _ = converted
    _, _, packet, images = prepared(root)
    packet["review_step"] = ReviewStep(
        name="content", kind="content", instruction="专门核对逻辑方向"
    ).model_dump()
    packet["current_review"] = [
        {"id": i, **item} for i, item in enumerate(packet["current_review"])
    ]
    valid = EditingProvider().review(packet, images)
    invalid = valid.model_copy(deep=True)
    # 内容轮试图改图须被拒绝，模型只能在独立修复请求中更正结果。
    invalid.frames = []
    if supplier == "deepseek":
        client = Client([response(invalid.model_dump_json()), response(valid.model_dump_json())])
        service = DeepSeekProvider(Config(max_calls=2), Events(root), client)
        service._history = [{"role": "assistant", "content": "旧生成历史"}]
    else:
        from test_qwen_provider import Client as QwenClient
        from test_qwen_provider import Stream, chunk

        from video_learner.providers.qwen import QwenProvider

        client = QwenClient(
            [
                Stream([chunk(content=value.model_dump_json(), finish="stop")])
                for value in (invalid, valid)
            ]
        )
        service = QwenProvider(Config(provider="qwen", max_calls=2), Events(root), client)
    assert service.review(packet, images) == valid
    assert len(client.requests) == 2
    assert all("旧生成历史" not in json.dumps(r, ensure_ascii=False) for r in client.requests)
    assert all("专门核对逻辑方向" in json.dumps(r, ensure_ascii=False) for r in client.requests)
    with pytest.raises(TaskError, match="上限"):
        service.review(packet, images)


def test_function_pipeline_uses_updated_state_and_keeps_unresolved_issues(converted, tmp_path):
    """自定义函数按配置顺序串联，两次文字插入不撞 ID；报告轮保留图片和既有疑点。"""
    root, _ = converted
    config_path = tmp_path / "review.toml"
    config_path.write_text(
        '[[review_steps]]\nname="visual"\nkind="visual"\n'
        '[[review_steps]]\nname="content"\nkind="content"\ninstruction="保留步骤"\n'
        '[[review_steps]]\nname="wording"\nkind="content"\n'
        '[[review_steps]]\nname="check"\nkind="content"\nmode="report"\n',
        encoding="utf-8",
    )
    calls = []

    def visual(packet, images):
        """换成末张候选并报告一条疑点，留给后续轮观察。"""
        calls.append(packet["review_step"]["name"])
        result = DeterministicProvider().review(packet, images)
        for frame in result.frames:
            frame.decision = "omit"
        result.frames[-1].decision = "use"
        result.findings = [
            ReviewFinding(
                target_id=packet["chapter"]["blocks"][0]["id"],
                kind="fidelity",
                reason="需要核对原课条件。",
                evidence_ids=[packet["transcript"][0]["id"]],
                action="report",
                blocks=[],
            )
        ]
        return result

    def content(packet, images):
        """观察前轮最新图块与正文，再插入一个有证据的文字块。"""
        name = packet["review_step"]["name"]
        calls.append(name)
        figures = [b for b in packet["chapter"]["blocks"] if b["kind"] == "figure"]
        assert len(figures) == 1 and figures[0]["frame_id"] == packet["frames"][-1]["id"]
        assert figures[0]["id"] in packet["current_markdown"]
        block = packet["chapter"]["blocks"][0]
        if name == "wording":
            assert "补充步骤 content" in packet["current_markdown"]
        return ReviewResult(
            frames=None,
            findings=[
                ReviewFinding(
                    target_id=block["id"],
                    kind="fidelity",
                    reason="保留原课操作步骤。",
                    evidence_ids=block["evidence_ids"],
                    action="insert_after",
                    blocks=[
                        DraftBlock(
                            kind="text",
                            body=f"补充步骤 {name}",
                            category="original",
                            evidence_ids=block["evidence_ids"],
                            frame_id=None,
                        )
                    ],
                )
            ],
            resolved_review_ids=[0] if name == "content" else [],
        )

    def check(packet, images):
        """报告模式不提出修改；忽略的已有疑点必须继续保留。"""
        calls.append(packet["review_step"]["name"])
        assert "补充步骤 wording" in packet["current_markdown"]
        assert [item["reason"] for item in packet["current_review"]] == ["需要核对原课条件。"]
        return ReviewResult(frames=None, findings=[])

    destination = review(
        root,
        config_path=config_path,
        reviewers={
            "visual": visual,
            "content": content,
            "wording": content,
            "check": check,
        },
    )
    assert calls == ["visual", "content", "wording", "check"]
    book = replay_revision(root, "r002")
    assert [item.reason for item in book.review if item.block_id] == ["需要核对原课条件。"]
    assert [item for item in book.review if item.block_id is None] == [
        item
        for item in Notebook.model_validate(read_json(root / "notes.json")).review
        if item.block_id is None
    ]
    blocks = book.chapters[0].blocks
    assert len({block.id for block in blocks}) == len(blocks) == 4
    assert len(image_dependencies((destination / "notes.md").read_bytes())) == 1
    changes = (destination / "changes.md").read_text(encoding="utf-8")
    assert all(f"## {name}" in changes for name in calls)
    with pytest.raises(InputError, match="名称"):
        review(root, reviewers={"unknown": check})


def test_functional_application_preserves_inputs_and_enforces_scope(converted):
    """纯应用不改输入；正文、图文和只报告模式不能越过各自的写入范围。"""
    root, _ = converted
    book, current, packet, images = prepared(root)
    packet["current_review"] = [{"id": i, **v} for i, v in enumerate(packet["current_review"])]
    packet["review_step"] = ReviewStep(name="content", kind="content").model_dump()
    result = EditingProvider().review(packet, images)
    before = book.model_dump_json(), json.dumps(packet), result.model_dump_json()
    updated, generated = apply_review_pass(book, packet, result, current)
    assert before == (book.model_dump_json(), json.dumps(packet), result.model_dump_json())
    assert generated != current and updated != book
    assert updated.review == book.review
    assert updated.chapters[0].blocks[1] == book.chapters[0].blocks[1]
    result.resolved_review_ids = [-1]
    with pytest.raises(TaskError, match="疑点"):
        apply_review_pass(book, packet, result, current)
    result.resolved_review_ids = []
    result.findings[0].target_id = book.chapters[0].blocks[1].id
    with pytest.raises(TaskError, match="只能修改文字"):
        reviewed_chapter(result, packet)
    result.findings[0].target_id = book.chapters[0].blocks[0].id
    packet["review_step"] = ReviewStep(name="visual", kind="visual").model_dump()
    result.frames = DeterministicProvider().review(packet, images).frames
    with pytest.raises(TaskError, match="只能替换"):
        reviewed_chapter(result, packet)
    packet["review_step"]["mode"] = "report"
    with pytest.raises(TaskError, match="只报告"):
        reviewed_chapter(result, packet)
    result.frames = None
    result.findings = []
    assert apply_review_pass(book, packet, result, current) == (book, current)


def test_visual_review_can_leave_existing_prose_defect_for_content_pass(converted):
    """旧讲义的正文缺陷应留到内容轮修复，图文轮与只报告轮不能被迫越权修改。"""
    root, _ = converted
    book, _, _, _ = prepared(root)
    book.chapters[0].blocks[0].body = "参见 " + book.frames[0].id
    current = render_notes(book)
    for step in (
        ReviewStep(name="visual", kind="visual"),
        ReviewStep(name="check", kind="content", mode="report"),
        ReviewStep(name="content", kind="content"),
    ):
        frozen, images = review_input(book, book.chapters[0], Config(), root, current, "r002", step)
        service = EditingProvider() if step.name == "content" else DeterministicProvider()
        result = service.review(frozen.packet, images)
        book, current = apply_review_pass(book, frozen.packet, result, current)
    assert "修正后的讲义说明" in current.decode()
    assert "参见 " + book.frames[0].id not in current.decode()


@pytest.mark.parametrize(
    "settings",
    [
        "review_steps=[]",
        'review_steps=[{name="same",kind="visual"},{name="same",kind="content"}]',
        'review_steps=[{name="../outside",kind="content"}]',
        'review_steps=[{name="con",kind="content"}]',
        'review_steps=[{name="code",kind="programming"}]',
    ],
)
def test_invalid_review_configuration_is_rejected(tmp_path, settings):
    """配置在付费请求前拒绝空流程、重复名称、路径名称及课程类型职能。"""
    path = tmp_path / "invalid.toml"
    path.write_text(settings, encoding="utf-8")
    with pytest.raises(InputError, match="配置无效"):
        load_config(path)
