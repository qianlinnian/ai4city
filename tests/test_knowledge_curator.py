from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from knowledge_base.knowledge_curator import (
    DeepSeekKnowledgeClient,
    KnowledgeBatch,
    PageText,
    batch_output_path,
    build_pdf_batches,
    build_curator_messages,
    discover_knowledge_pdfs,
    process_batch,
    payload_status,
    match_evidence,
    infer_line_ids,
    validate_curator_output,
    write_batch_draft,
)
from scripts.build_curated_knowledge import (
    draft_is_empty,
    draft_requires_api_retry,
    select_shard,
)


ROOT = Path(__file__).resolve().parents[1]


def _payload(*, quote: str, metric_links: list[str] | None = None) -> str:
    return json.dumps(
        {
            "items": [
                {
                    "section": "园路",
                    "clause_id": "5.1.4",
                    "title": "主要园路连续通行",
                    "summary": "主要园路应保持连续通行。",
                    "normative_level": "mandatory",
                    "task_scope": ["task3"],
                    "scene_types": ["公园"],
                    "objects": ["园路"],
                    "actions": ["保持连续通行"],
                    "constraints": ["避免阻塞"],
                    "prohibited_actions": [],
                    "metric_links": metric_links or ["built_ratio"],
                    "keywords": ["园路", "通行"],
                    "evidence": [{"page": 7, "quote": quote}],
                    "confidence": 0.92,
                    "needs_review": False,
                }
            ]
        },
        ensure_ascii=False,
    )


class _FakeResponse:
    def __init__(
        self,
        content: str | None,
        *,
        finish_reason: str = "stop",
        reasoning_content: str | None = None,
    ) -> None:
        self.choices = [
            type(
                "Choice",
                (),
                {
                    "finish_reason": finish_reason,
                    "message": type(
                        "Message",
                        (),
                        {
                            "content": content,
                            "reasoning_content": reasoning_content,
                        },
                    )(),
                },
            )()
        ]


class _FakeCompletions:
    def __init__(self, responses: dict[str, str | Exception | _FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses[kwargs["model"]]
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, _FakeResponse) else _FakeResponse(response)


class _FakeClient:
    def __init__(self, responses: dict[str, str | Exception | _FakeResponse]) -> None:
        self.chat = type("Chat", (), {})()
        self.chat.completions = _FakeCompletions(responses)


class KnowledgeCuratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = ROOT / ".cache" / "test_knowledge_curator" / uuid.uuid4().hex
        self.root.mkdir(parents=True)
        self.source = self.root / "公园设计规范.pdf"
        self.source.write_bytes(b"fake-pdf")
        self.batch = KnowledgeBatch(
            source=self.source.resolve(),
            source_fingerprint="abc123",
            batch_id="batch-0001",
            pages=(PageText(7, "主要园路应保持连续通行，并避免设施阻塞。"),),
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_discovery_includes_all_pdfs(self) -> None:
        first = self.root / "城市规范.pdf"
        second = self.root / "空间设计指南.pdf"
        first.write_bytes(b"first")
        second.write_bytes(b"second")
        paths = discover_knowledge_pdfs(self.root)
        self.assertCountEqual(
            paths,
            [first.resolve(), second.resolve(), self.source.resolve()],
        )

    def test_shards_are_disjoint_and_cover_all_batches(self) -> None:
        items = list(range(17))
        shards = [select_shard(items, 4, index) for index in range(4)]
        self.assertEqual(sorted(item for shard in shards for item in shard), items)
        self.assertEqual(
            sum(len(set(left) & set(right)) for left in shards for right in shards if left is not right),
            0,
        )
        with self.assertRaises(ValueError):
            select_shard(items, 4, 4)

    def test_pdf_batches_preserve_page_numbers(self) -> None:
        pages = [
            "第一页有足够长的规范文字，用于描述公园入口和道路连续通行要求。" * 3,
            "第二页有足够长的规范文字，用于描述绿化和基础设施保护要求。" * 3,
        ]
        with patch.object(
            __import__(
                "knowledge_base.knowledge_curator", fromlist=["LocalTfidfRagProvider"]
            ).LocalTfidfRagProvider,
            "_extract_pdf_pages",
            return_value=pages,
        ):
            batches = build_pdf_batches(self.source, max_chars=1000, max_pages=1)
        self.assertEqual([batch.page_numbers for batch in batches], [[1], [2]])
        _, user = build_curator_messages(batches[0])
        self.assertIn("[P0001-L0001]", user)

    def test_validation_requires_evidence_and_metric_whitelist(self) -> None:
        valid = validate_curator_output(
            self.batch, _payload(quote="主要园路应保持连续通行")
        )
        self.assertFalse(valid.requires_review)
        self.assertEqual(valid.records[0]["review_status"], "program_validated")

        invalid = validate_curator_output(
            self.batch,
            _payload(quote="原文不存在的句子", metric_links=["enclosure"]),
        )
        self.assertTrue(invalid.requires_review)
        self.assertIn("非法形态指标", " ".join(invalid.validation_errors))
        self.assertIn("无法在第 7 页原文中定位", " ".join(invalid.validation_errors))

        empty = validate_curator_output(self.batch, '{"items": []}')
        self.assertEqual(payload_status(empty), "empty")
        invalid_json = validate_curator_output(self.batch, '{"items": [')
        self.assertEqual(payload_status(invalid_json), "needs_review")
        self.assertEqual(
            valid.records[0]["evidence"][0]["line_ids"], ["P0007-L0001"]
        )

    def test_ocr_duplicate_lines_can_use_auditable_fuzzy_match(self) -> None:
        page = (
            "本规范适用于城乡各类公园的新建、扩建、改建和修复\n"
            "本规泡适用于城乡各类公园的新建、打建、改建和修\n"
            "的设计。\n的设计。"
        )
        match = match_evidence(
            "本规范适用于城乡各类公园的新建、扩建、改建和修复的设计。",
            page,
        )
        self.assertEqual(match.match_type, "fuzzy_ocr")
        self.assertGreaterEqual(match.score, 0.85)
        self.assertEqual(match_evidence("凭空添加水体", page).match_type, "missing")
        lines = {
            "P0010-L0001": "本规范适用于城乡各类公园的新建、扩建、改建和修复",
            "P0010-L0002": "本规泡适用于城乡各类公园的新建、打建、改建和修",
            "P0010-L0003": "的设计。",
        }
        inferred = infer_line_ids(
            "本规范适用于城乡各类公园的新建、扩建、改建和修复的设计。",
            lines,
        )
        self.assertEqual(inferred, ["P0010-L0001", "P0010-L0003"])

    def test_auto_pro_retries_only_when_flash_requires_review(self) -> None:
        fake = _FakeClient(
            {
                "flash": _payload(quote="原文不存在的句子"),
                "pro": _payload(quote="主要园路应保持连续通行"),
            }
        )
        client = DeepSeekKnowledgeClient(
            api_key="test-key",
            base_url="https://example.invalid",
            flash_model="flash",
            pro_model="pro",
            client=fake,
        )
        model, tier, result = process_batch(client, self.batch, auto_pro=True)
        self.assertEqual((model, tier), ("pro", "pro"))
        self.assertFalse(result.requires_review)
        self.assertEqual(
            [call["model"] for call in fake.chat.completions.calls], ["flash", "pro"]
        )
        self.assertEqual(
            fake.chat.completions.calls[0]["response_format"],
            {"type": "json_object"},
        )
        self.assertEqual(
            fake.chat.completions.calls[0]["extra_body"],
            {"thinking": {"type": "disabled"}},
        )
        self.assertEqual(fake.chat.completions.calls[0]["max_tokens"], 16000)

    def test_non_stop_or_reasoning_only_response_is_rejected(self) -> None:
        for response in (
            _FakeResponse('{"items": [', finish_reason="length"),
            _FakeResponse(None, reasoning_content="still reasoning"),
        ):
            fake = _FakeClient({"flash": response, "pro": response})
            client = DeepSeekKnowledgeClient(
                api_key="test-key",
                base_url="https://example.invalid",
                flash_model="flash",
                pro_model="pro",
                client=fake,
            )
            with self.assertRaises(ValueError):
                client.curate(self.batch)

    def test_retry_invalid_only_selects_drafts_with_validation_errors(self) -> None:
        valid = self.root / "valid.json"
        invalid = self.root / "invalid.json"
        valid.write_text('{"review_status":"empty","validation_errors":[]}', encoding="utf-8")
        invalid.write_text(
            '{"review_status":"needs_review","validation_errors":["truncated"]}',
            encoding="utf-8",
        )
        self.assertFalse(draft_requires_api_retry(valid))
        self.assertTrue(draft_requires_api_retry(invalid))
        self.assertTrue(draft_is_empty(valid))
        self.assertFalse(draft_is_empty(invalid))

    def test_auto_pro_recovers_from_flash_api_failure(self) -> None:
        fake = _FakeClient(
            {
                "flash": RuntimeError("temporary failure"),
                "pro": _payload(quote="主要园路应保持连续通行"),
            }
        )
        client = DeepSeekKnowledgeClient(
            api_key="test-key",
            base_url="https://example.invalid",
            flash_model="flash",
            pro_model="pro",
            client=fake,
        )
        model, tier, result = process_batch(client, self.batch, auto_pro=True)
        self.assertEqual((model, tier), ("pro", "pro"))
        self.assertFalse(result.requires_review)

    def test_draft_is_written_as_rag_compatible_records(self) -> None:
        result = validate_curator_output(
            self.batch, _payload(quote="主要园路应保持连续通行")
        )
        output = batch_output_path(self.root / "drafts", self.batch)
        write_batch_draft(
            output,
            batch=self.batch,
            model="flash",
            tier="flash",
            payload=result,
        )
        data = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], "ai4city-knowledge-draft-v2")
        self.assertTrue(data["records"])
        self.assertEqual(data["records"][0]["source"], str(self.source.resolve()))


if __name__ == "__main__":
    unittest.main()
