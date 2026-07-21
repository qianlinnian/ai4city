from __future__ import annotations

import json
import shutil
import subprocess
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agents.cartographer_agent import CartographerAgent
from agents.translator_agent import TranslatorAgent
from config import EXPERIENCE_KEYS
from knowledge_base.kb_store import KnowledgeBase
from knowledge_base.rag_provider import (
    LocalTfidfRagProvider,
    NullRagProvider,
    QwenEmbeddingRagProvider,
    build_default_rag_provider,
)


ROOT = Path(__file__).resolve().parents[1]


class FakeEmbeddingClient:
    """不联网的 DashScope OpenAI-compatible Embedding 客户端替身。"""

    def __init__(self) -> None:
        self.embeddings = self
        self.calls: list[list[str]] = []

    def create(self, *, input, dimensions, **_kwargs):
        texts = list(input)
        self.calls.append(texts)
        data = []
        for index, text in enumerate(texts):
            vector = [0.0] * dimensions
            vector[0 if "社区" in text or "入口" in text else 1] = 1.0
            data.append(SimpleNamespace(index=index, embedding=vector))
        return SimpleNamespace(data=data)


class FailingEmbeddingClient:
    def __init__(self) -> None:
        self.embeddings = self

    def create(self, **_kwargs):
        raise RuntimeError("simulated embedding outage")


class RagProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = ROOT / ".cache" / "test_rag_provider" / uuid.uuid4().hex
        self.root.mkdir(parents=True)
        self.knowledge = self.root / "rules.txt"
        self.knowledge.write_text(
            "临街停留区应保持道路通行，并使用乔灌草复层绿化。\n"
            "全景左右接缝附近不得放置重复对象。",
            encoding="utf-8",
        )
        self.baseline = {
            "green_view": 0.2,
            "blue_view": 0.05,
            "sky_view": 0.3,
            "built_ratio": 0.5,
            "color_richness": 5.0,
            "edge_density": 0.08,
            "skyline_variance": 0.02,
        }
        self.experience = {key: 3.0 for key in EXPERIENCE_KEYS}
        self.targets = {**self.experience, "comfort": 4.0}

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_disabled_provider_does_not_build_index(self) -> None:
        provider = LocalTfidfRagProvider([self.knowledge], enabled=False)
        self.assertEqual(provider.retrieve(experience_records=[]), [])
        self.assertEqual(provider.indexed_chunk_count, 0)
        self.assertFalse(NullRagProvider().enabled)

        class DisabledSpy:
            enabled = False

            def retrieve(self, **_kwargs):
                raise AssertionError("关闭的 RAG 不应执行 retrieve")

        with patch("agents.translator_agent.llm_client.chat", return_value=None):
            result = TranslatorAgent(
                KnowledgeBase(self.root / "disabled-kb"), DisabledSpy()
            ).run(
                experience_records=[
                    {"person_id": "p1", "experience": self.experience}
                ],
                experience_targets=self.targets,
                baseline_metrics=self.baseline,
            )
        self.assertFalse(result.references_used)

    def test_enabled_local_retrieval_returns_required_fields(self) -> None:
        provider = LocalTfidfRagProvider(
            [self.knowledge], top_k=2, min_score=0.0, chunk_size=200
        )
        results = provider.retrieve(
            baseline_metrics=self.baseline,
            target_metrics={**self.baseline, "green_view": 0.3},
            scene_context="临街停留区",
            expert_advice="保持道路",
            scene_understanding={},
        )
        self.assertTrue(results)
        self.assertEqual(
            {"text", "source", "chunk_id", "score", "metadata"},
            set(results[0]) - {"id"},
        )
        self.assertIn("道路", results[0]["text"])

    def test_qwen_embedding_retrieval_uses_cache_and_marks_result_mode(self) -> None:
        community = self.root / "community.txt"
        community.write_text("社区入口的休憩设施应保持消防通行。", encoding="utf-8")
        other = self.root / "other.txt"
        other.write_text("商业步行街的导视系统应保持连续。", encoding="utf-8")
        cache_dir = self.root / "qwen-cache"
        client = FakeEmbeddingClient()
        provider = QwenEmbeddingRagProvider(
            [community, other],
            api_key="test-key-never-sent",
            dimensions=3,
            embedding_client=client,
            cache_dir=cache_dir,
            top_k=2,
            min_score=0.0,
        )
        results = provider.retrieve(
            baseline_metrics=self.baseline,
            target_metrics=self.baseline,
            scene_context="老旧社区入口",
            expert_advice="保持消防通行",
            scene_understanding={},
        )
        self.assertTrue(results)
        self.assertEqual(results[0]["metadata"]["retrieval_mode"], "qwen_embedding")
        self.assertIn("社区入口", results[0]["text"])
        self.assertEqual(len(client.calls), 2)  # 一次文档批量 + 一次查询
        self.assertTrue(list((cache_dir / "qwen_embeddings").glob("*.json")))

        cached_client = FakeEmbeddingClient()
        cached_provider = QwenEmbeddingRagProvider(
            [community, other],
            api_key="another-test-key-never-sent",
            dimensions=3,
            embedding_client=cached_client,
            cache_dir=cache_dir,
            top_k=2,
            min_score=0.0,
        )
        cached_provider.retrieve(
            baseline_metrics=self.baseline,
            target_metrics=self.baseline,
            scene_context="老旧社区入口",
            expert_advice="保持消防通行",
            scene_understanding={},
        )
        self.assertEqual(len(cached_client.calls), 1)  # 文档向量命中磁盘缓存

    def test_dynamic_feedback_uses_incremental_embedding_cache(self) -> None:
        static_source = self.root / "static.txt"
        static_source.write_text("社区入口应保持消防通行与连续步行空间。", encoding="utf-8")
        feedback_source = self.root / "learning_feedback.json"
        feedbacks = [
            {"id": "feedback-1", "notes": "社区入口座椅不得阻塞消防通道"}
        ]
        feedback_source.write_text(
            json.dumps({"feedbacks": feedbacks}, ensure_ascii=False), encoding="utf-8"
        )
        cache_dir = self.root / "incremental-cache"
        first_client = FakeEmbeddingClient()
        first_provider = QwenEmbeddingRagProvider(
            [static_source],
            dynamic_source_paths=[feedback_source],
            api_key="test-key-never-sent",
            dimensions=3,
            embedding_client=first_client,
            cache_dir=cache_dir,
            top_k=2,
            min_score=0.0,
        )
        first_results = first_provider.retrieve(
            baseline_metrics=self.baseline,
            target_metrics=self.baseline,
            scene_context="社区入口",
            expert_advice="保持消防通行",
            scene_understanding={},
        )
        self.assertEqual([len(batch) for batch in first_client.calls], [1, 1, 1])
        self.assertTrue(
            any(
                item["metadata"].get("source_group") == "dynamic_feedback"
                for item in first_results
            )
        )

        feedbacks.append(
            {"id": "feedback-2", "notes": "社区入口绿化不应遮挡安全视线"}
        )
        feedback_source.write_text(
            json.dumps({"feedbacks": feedbacks}, ensure_ascii=False), encoding="utf-8"
        )
        second_client = FakeEmbeddingClient()
        second_provider = QwenEmbeddingRagProvider(
            [static_source],
            dynamic_source_paths=[feedback_source],
            api_key="test-key-never-sent",
            dimensions=3,
            embedding_client=second_client,
            cache_dir=cache_dir,
            top_k=2,
            min_score=0.0,
        )
        second_provider.retrieve(
            baseline_metrics=self.baseline,
            target_metrics=self.baseline,
            scene_context="社区入口",
            expert_advice="保持消防通行",
            scene_understanding={},
        )
        # Static documents stay cached; only the appended feedback plus query embed.
        self.assertEqual([len(batch) for batch in second_client.calls], [1, 1])

    def test_qwen_embedding_failure_falls_back_to_tfidf(self) -> None:
        provider = QwenEmbeddingRagProvider(
            [self.knowledge],
            api_key="test-key-never-sent",
            dimensions=3,
            embedding_client=FailingEmbeddingClient(),
            cache_dir=self.root / "qwen-fallback-cache",
            min_score=0.0,
        )
        results = provider.retrieve(
            baseline_metrics=self.baseline,
            target_metrics=self.baseline,
            scene_context="街道停留区",
            expert_advice="保持道路",
            scene_understanding={},
        )
        self.assertTrue(results)
        self.assertEqual(results[0]["metadata"]["retrieval_mode"], "tfidf_fallback")
        self.assertIn("simulated embedding outage", provider.last_embedding_error)

    def test_default_provider_selects_qwen_embedding_when_configured(self) -> None:
        with (
            patch("knowledge_base.rag_provider.RAG_ENABLED", True),
            patch("knowledge_base.rag_provider.RAG_RETRIEVAL_MODE", "qwen_embedding"),
        ):
            self.assertIsInstance(build_default_rag_provider(), QwenEmbeddingRagProvider)

    def test_task3_scene_cases_are_weighted_limited_and_excluded_from_task2(self) -> None:
        case_file = self.root / "scene_prompt_examples.json"
        records = []
        for index in range(1, 6):
            records.append(
                {
                    "id": f"community-{index}",
                    "case_type": "task3_scene_prompt_example",
                    "scene_type": "community",
                    "scene_label": "社区",
                    "retrieval_text": "老旧社区入口 停车秩序 消防通行 日常座椅",
                    "review_status": "program_validated",
                }
            )
        records.append(
            {
                "id": "office-1",
                "case_type": "task3_scene_prompt_example",
                "scene_type": "commercial_office",
                "scene_label": "商办",
                "retrieval_text": "商办街区 办公入口 配送 外摆",
                "review_status": "program_validated",
            }
        )
        case_file.write_text(
            json.dumps({"records": records}, ensure_ascii=False), encoding="utf-8"
        )
        provider = LocalTfidfRagProvider(
            [case_file, self.knowledge], top_k=5, min_score=0.0
        )
        task3_results = provider.retrieve(
            baseline_metrics=self.baseline,
            target_metrics=self.baseline,
            scene_context="老旧社区入口与住宅停车",
            expert_advice="保持消防通行",
            scene_understanding={},
        )
        case_results = [
            result
            for result in task3_results
            if result["metadata"].get("case_type")
            == "task3_scene_prompt_example"
        ]
        self.assertTrue(case_results)
        self.assertLessEqual(len(case_results), 3)
        self.assertEqual(case_results[0]["metadata"]["scene_type"], "community")

        task2_results = provider.retrieve(
            experience_records=[{"person_id": "p1", "experience": self.experience}],
            experience_targets=self.targets,
            baseline_metrics=self.baseline,
            scene_context="老旧社区入口",
        )
        self.assertFalse(
            any(
                result["metadata"].get("case_type")
                == "task3_scene_prompt_example"
                for result in task2_results
            )
        )

    def test_pdf_is_extracted_by_page_and_cached_outside_data_source(self) -> None:
        pdf = self.root / "规范资料.pdf"
        pdf.write_bytes(b"fake-pdf-fixture")
        extracted = (
            "第一页 公园道路与主要出入口应保持连续通行，避免占用消防通道，"
            "并确保步行空间具有足够宽度和清晰导向。"
            "\f第二页 绿化配置应保持视线安全并保护既有基础设施，"
            "不得遮挡交通标识、消防设施和建筑入口。"
        ).encode("utf-8")
        completed = subprocess.CompletedProcess(
            args=["pdftotext"], returncode=0, stdout=extracted, stderr=b""
        )
        cache_dir = self.root / "rag-cache"
        provider = LocalTfidfRagProvider(
            [pdf], top_k=2, min_score=0.0, cache_dir=cache_dir
        )
        with (
            patch("knowledge_base.rag_provider.shutil.which", return_value="pdftotext"),
            patch("knowledge_base.rag_provider.subprocess.run", return_value=completed),
        ):
            results = provider.retrieve(
                baseline_metrics=self.baseline,
                target_metrics=self.baseline,
                scene_context="公园道路",
                expert_advice="保持消防通道",
                scene_understanding={},
            )
        self.assertTrue(results)
        self.assertIn(results[0]["metadata"]["page"], {1, 2})
        self.assertEqual(results[0]["metadata"]["format"], "pdf")
        self.assertTrue(list(cache_dir.glob("*.json")))

    def test_default_sources_include_all_knowledge_files(self) -> None:
        knowledge_dir = self.root / "knowledge"
        knowledge_dir.mkdir()
        first = knowledge_dir / "公园设计规范.pdf"
        second = knowledge_dir / "空间设计指南.pdf"
        first.write_bytes(b"first")
        second.write_bytes(b"second")
        with (
            patch("knowledge_base.rag_provider.KNOWLEDGE_SOURCE_DIR", knowledge_dir),
            patch(
                "knowledge_base.rag_provider.RAG_PUBLISHED_KNOWLEDGE_DIR",
                self.root / "empty-published",
            ),
            patch("knowledge_base.rag_provider.RAG_INCLUDE_REPOSITORY_SOURCES", False),
        ):
            paths = LocalTfidfRagProvider.default_source_paths()
        self.assertEqual(paths, [first, second])

    def test_empty_retrieval_and_llm_failure_do_not_break_task2_or_task3(self) -> None:
        provider = LocalTfidfRagProvider(
            [self.knowledge], min_score=1.1
        )
        kb_dir = self.root / "kb"
        kb = KnowledgeBase(kb_dir)
        with (
            patch("agents.translator_agent.llm_client.chat", return_value=None),
            patch("agents.cartographer_agent.llm_client.chat", return_value=None),
        ):
            translation = TranslatorAgent(kb, provider).run(
                experience_records=[
                    {"person_id": "p1", "experience": self.experience}
                ],
                experience_targets=self.targets,
                baseline_metrics=self.baseline,
                scene_understanding={"status": "degraded"},
            )
            plan = CartographerAgent(kb, provider).run(
                baseline_metrics=self.baseline,
                target_metrics=translation.target_metrics.as_dict(),
                scene_understanding={"status": "degraded"},
                language="zh",
            )
        self.assertFalse(translation.references_used)
        self.assertFalse(plan.rag_references)
        self.assertTrue(plan.draft_text)


if __name__ == "__main__":
    unittest.main()
