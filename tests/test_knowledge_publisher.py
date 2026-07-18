from __future__ import annotations

import hashlib
import json
import shutil
import unittest
import uuid
from pathlib import Path

from knowledge_base.knowledge_publisher import (
    PublishPolicy,
    collect_publication_records,
    write_publication,
)


ROOT = Path(__file__).resolve().parents[1]


class KnowledgePublisherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = ROOT / ".cache" / "test_knowledge_publisher" / uuid.uuid4().hex
        self.drafts = self.root / "drafts"
        self.drafts.mkdir(parents=True)
        self.source = self.root / "规范.pdf"
        self.source.write_bytes(b"source")
        stat = self.source.stat()
        self.fingerprint = hashlib.sha256(
            f"{self.source.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
        ).hexdigest()

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def record(self, knowledge_id: str, **overrides):
        record = {
            "knowledge_id": knowledge_id,
            "source": str(self.source),
            "title": "保持道路连续通行",
            "summary": "主要道路应保持连续通行。",
            "metric_links": ["built_ratio"],
            "confidence": 0.95,
            "needs_review": False,
            "high_risk_normative_text": False,
            "review_status": "program_validated",
            "validation_errors": [],
            "evidence": [
                {
                    "page": 3,
                    "line_ids": ["P0003-L0002"],
                    "quote": "主要道路应保持连续通行",
                    "match_type": "exact",
                    "match_score": 1.0,
                }
            ],
        }
        record.update(overrides)
        return record

    def write_draft(self, records):
        path = self.drafts / "batch-0001.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": "ai4city-knowledge-draft-v2",
                    "source": str(self.source),
                    "source_fingerprint": self.fingerprint,
                    "records": records,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def test_only_program_validated_records_are_published(self) -> None:
        safe = self.record("safe")
        high_risk = self.record(
            "high-risk",
            review_status="needs_review",
            high_risk_normative_text=True,
        )
        self.write_draft([safe, high_risk])
        published, held = collect_publication_records(self.drafts)
        self.assertEqual([item["knowledge_id"] for item in published], ["safe"])
        self.assertEqual([item["knowledge_id"] for item in held], ["high-risk"])

        self.assertEqual(published[0]["publication"]["approval_type"], "program_gate")

    def test_hard_evidence_error_cannot_be_overridden(self) -> None:
        broken = self.record(
            "broken",
            evidence=[
                {
                    "page": 3,
                    "line_ids": [],
                    "quote": "无法定位",
                    "match_type": "missing",
                    "match_score": 0.0,
                }
            ],
        )
        self.write_draft([broken])
        published, held = collect_publication_records(self.drafts)
        self.assertFalse(published)
        self.assertTrue(held[0]["hard_gate_errors"])

    def test_low_fuzzy_score_is_held_and_publication_is_written(self) -> None:
        low = self.record(
            "low",
            evidence=[
                {
                    "page": 3,
                    "line_ids": ["P0003-L0002"],
                    "quote": "道路通行",
                    "match_type": "fuzzy_ocr",
                    "match_score": 0.88,
                }
            ],
        )
        safe = self.record("safe")
        self.write_draft([low, safe])
        published, held = collect_publication_records(
            self.drafts, policy=PublishPolicy(min_fuzzy_score=0.9)
        )
        self.assertEqual([item["knowledge_id"] for item in published], ["safe"])
        self.assertEqual([item["knowledge_id"] for item in held], ["low"])
        paths = write_publication(published, output_dir=self.root / "published")
        data = json.loads(paths[0].read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], "ai4city-rag-knowledge-v1")
        self.assertEqual(data["record_count"], 1)
        self.assertEqual(data["source"], self.source.name)
        self.assertEqual(data["records"][0]["source"], self.source.name)
        self.assertNotIn(str(self.root), paths[0].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
