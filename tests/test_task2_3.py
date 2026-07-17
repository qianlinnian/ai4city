from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from agents.cartographer_agent import CartographerAgent
from agents.translator_agent import TranslatorAgent
from config import EXPERIENCE_KEYS, MORPH_BOUNDS, MORPH_KEYS
from knowledge_base.kb_store import KnowledgeBase
from schemas.models import ExperienceTargets, SceneContext


ROOT = Path(__file__).resolve().parents[1]


class Task23TestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.data_dir = ROOT / ".cache" / "test_task2_3" / uuid.uuid4().hex
        self.data_dir.mkdir(parents=True)
        shutil.copy(ROOT / "knowledge_base" / "data" / "mapping_rules.json", self.data_dir)
        shutil.copy(
            ROOT / "knowledge_base" / "data" / "experience_morph_cases.json",
            self.data_dir,
        )
        self.kb = KnowledgeBase(data_dir=self.data_dir)
        self.baseline_metrics = {
            "green_view": 0.20,
            "blue_view": 0.08,
            "sky_view": 0.25,
            "built_ratio": 0.50,
            "edge_density": 0.09,
            "color_richness": 5.0,
            "skyline_variance": 0.02,
        }
        self.experience_baseline = {key: 3.0 for key in EXPERIENCE_KEYS}
        self.experience_targets = {
            "comfort": 4.0,
            "naturalness": 4.0,
            "safety": 4.0,
            "relaxation": 4.0,
            "environmental_disturbance": 2.0,
            "stay_intention": 4.0,
            "overall_impression": 4.0,
        }
        self.experience_records = [
            {
                "person_id": "student-1",
                "person_name": "同学甲",
                "experience": {**self.experience_baseline, "comfort": 2.0},
            },
            {
                "person_id": "student-2",
                "person_name": "同学乙",
                "experience": {
                    **self.experience_baseline,
                    "naturalness": 4.0,
                    "environmental_disturbance": 4.0,
                },
            },
        ]

    def tearDown(self) -> None:
        cache_root = (ROOT / ".cache" / "test_task2_3").resolve()
        target = self.data_dir.resolve()
        if target.is_relative_to(cache_root):
            shutil.rmtree(target, ignore_errors=True)

    def test_seven_experience_fields_and_legacy_aliases(self) -> None:
        model = ExperienceTargets(
            comfort=4,
            naturalness=3,
            restoration=4.5,
            safety=4,
            environmental_disturbance=3,
            pleasure=4.2,
            stay=4.1,
        )
        values = model.as_dict()

        self.assertEqual(list(values), EXPERIENCE_KEYS)
        self.assertEqual(values["relaxation"], 4.5)
        self.assertEqual(values["overall_impression"], 4.2)
        self.assertEqual(values["stay_intention"], 4.1)
        self.assertEqual(values["naturalness"], 3.0)
        self.assertEqual(values["environmental_disturbance"], 3.0)

    def test_experience_scores_are_complete_and_reject_six(self) -> None:
        with self.assertRaises(ValidationError):
            ExperienceTargets(
                **{**self.experience_targets, "environmental_disturbance": 6}
            )
        incomplete = dict(self.experience_targets)
        incomplete.pop("overall_impression")
        with self.assertRaises(ValidationError):
            ExperienceTargets(**incomplete)

        invalid_records = [
            {
                "person_id": "student-with-typo",
                "experience": {**self.experience_baseline, "comfort": 6},
            }
        ]
        with self.assertRaisesRegex(ValueError, "student-with-typo"):
            TranslatorAgent(knowledge_base=self.kb).run(
                experience_records=invalid_records,
                experience_targets=self.experience_targets,
                baseline_metrics=self.baseline_metrics,
            )

    def test_parameter_names_and_legacy_scene_aliases(self) -> None:
        self.assertEqual(len(EXPERIENCE_KEYS), 7)
        self.assertEqual(len(MORPH_KEYS), 7)
        self.assertEqual(
            MORPH_KEYS,
            [
                "green_view",
                "blue_view",
                "sky_view",
                "built_ratio",
                "color_richness",
                "edge_density",
                "skyline_variance",
            ],
        )
        scene = SceneContext(
            location_type="社区",
            time_of_day="午后",
            weather="晴",
            crowd_level="中",
            sound_type="自然声",
            maintenance_status="良好",
            traffic_flow="低",
        )
        self.assertEqual(scene.space_type, "社区")
        self.assertEqual(scene.observation_time, "午后")
        self.assertEqual(scene.observation_weather, "晴")
        self.assertEqual(scene.people_flow, "中")
        self.assertIn("管理维护状态: 良好", scene.as_text())

    def test_task2_rule_fallback_keeps_all_people_and_does_not_use_rag(self) -> None:
        translator = TranslatorAgent(knowledge_base=self.kb)
        with (
            patch("agents.translator_agent.llm_client.chat", return_value=None),
            patch("agents.translator_agent.llm_client.chat_with_image", return_value=None),
            patch.object(
                self.kb,
                "retrieve_experience_cases",
                side_effect=AssertionError("Task 2当前不应调用RAG"),
            ),
        ):
            result = translator.run(
                experience_records=self.experience_records,
                experience_targets=self.experience_targets,
                baseline_metrics=self.baseline_metrics,
                scene_context="城市口袋公园；午后",
                original_image_path=str(ROOT / "data" / "p1.jpg"),
            )

        self.assertEqual(set(result.experience_targets), set(EXPERIENCE_KEYS))
        self.assertEqual(len(result.experience_records), 2)
        self.assertGreater(result.target_metrics.green_view, self.baseline_metrics["green_view"])
        self.assertLess(result.target_metrics.built_ratio, self.baseline_metrics["built_ratio"])
        self.assertLess(result.target_metrics.edge_density, self.baseline_metrics["edge_density"])
        self.assertFalse(result.references_used)
        self.assertIn("规则兜底", result.rationale)
        methods = {item.method for item in result.conversion_basis}
        self.assertEqual(methods, {"rule"})

    def test_task2_prompt_receives_every_person_and_returns_only_targets(self) -> None:
        model_target = {
            "green_view": 0.31,
            "blue_view": 0.09,
            "sky_view": 0.28,
            "built_ratio": 0.43,
            "color_richness": 7.0,
            "edge_density": 0.07,
            "skyline_variance": 0.025,
        }
        with (
            patch(
                "agents.translator_agent.llm_client.chat_with_image",
                return_value=json.dumps(model_target),
            ) as mocked_chat,
            patch.object(
                self.kb,
                "retrieve_experience_cases",
                side_effect=AssertionError("Task 2当前不应调用RAG"),
            ),
        ):
            result = TranslatorAgent(knowledge_base=self.kb).run(
                experience_records=self.experience_records,
                experience_targets=self.experience_targets,
                baseline_metrics=self.baseline_metrics,
                scene_context="蓝绿社区空间",
                original_image_path=str(ROOT / "data" / "p1.jpg"),
            )

        prompt_text = mocked_chat.call_args.args[1]
        self.assertIn("student-1", prompt_text)
        self.assertIn("student-2", prompt_text)
        self.assertIn('"baseline_metrics"', prompt_text)
        self.assertEqual(result.target_metrics.as_dict(), model_target)
        self.assertEqual({item.method for item in result.conversion_basis}, {"llm"})
        self.assertFalse(result.references_used)

    def test_task2_rejects_incomplete_or_invalid_morph_baseline(self) -> None:
        translator = TranslatorAgent(knowledge_base=self.kb)
        incomplete = dict(self.baseline_metrics)
        incomplete.pop("sky_view")
        with self.assertRaisesRegex(ValueError, "缺少指标: sky_view"):
            translator.run(
                experience_baseline=self.experience_baseline,
                experience_targets=self.experience_targets,
                baseline_metrics=incomplete,
            )

        invalid = {**self.baseline_metrics, "green_view": 1.2}
        with self.assertRaisesRegex(ValueError, "green_view必须位于0到1之间"):
            translator.run(
                experience_baseline=self.experience_baseline,
                experience_targets=self.experience_targets,
                baseline_metrics=invalid,
            )

        invalid_color = {**self.baseline_metrics, "color_richness": 24.1}
        with self.assertRaisesRegex(ValueError, "color_richness必须位于0到24之间"):
            translator.run(
                experience_baseline=self.experience_baseline,
                experience_targets=self.experience_targets,
                baseline_metrics=invalid_color,
            )

    def test_morph_bounds_follow_formula_theoretical_ranges(self) -> None:
        self.assertEqual(
            MORPH_BOUNDS,
            {
                "green_view": (0.0, 1.0),
                "blue_view": (0.0, 1.0),
                "sky_view": (0.0, 1.0),
                "built_ratio": (0.0, 1.0),
                "edge_density": (0.0, 1.0),
                "color_richness": (0.0, 24.0),
                "skyline_variance": (0.0, 1.0),
            },
        )

        formula_edge_values = {
            "green_view": 1.0,
            "blue_view": 1.0,
            "sky_view": 1.0,
            "built_ratio": 1.0,
            "edge_density": 1.0,
            "color_richness": 24.0,
            "skyline_variance": 1.0,
        }
        TranslatorAgent(knowledge_base=self.kb)._normalize_baseline(
            formula_edge_values
        )

    def test_task3_returns_structured_layout_and_edit_text(self) -> None:
        cartographer = CartographerAgent(knowledge_base=self.kb)
        target_metrics = {
            **self.baseline_metrics,
            "green_view": 0.38,
            "built_ratio": 0.42,
            "edge_density": 0.05,
            "color_richness": 6.0,
        }
        with (
            patch("agents.cartographer_agent.llm_client.chat", return_value=None),
            patch("agents.cartographer_agent.llm_client.chat_with_image", return_value=None),
            patch.object(
                self.kb,
                "retrieve_experience_cases",
                side_effect=AssertionError("Task 3当前不应调用RAG"),
            ),
        ):
            plan = cartographer.run(
                baseline_metrics=self.baseline_metrics,
                target_metrics=target_metrics,
                experience_baseline=self.experience_baseline,
                experience_targets=self.experience_targets,
                scene_context="城市口袋公园；午后",
                original_image_path=str(ROOT / "data" / "p1.jpg"),
                expert_advice="保留左侧建筑立面，优先改善右侧停留区",
                language="zh",
            )

        self.assertTrue(plan.plan_summary)
        self.assertTrue(plan.object_actions)
        self.assertTrue(all(action.position for action in plan.object_actions))
        self.assertTrue(plan.spatial_relations)
        self.assertTrue(plan.unchanged_regions)
        self.assertTrue(plan.constraints)
        self.assertIn("保持原有360°全景几何结构", plan.draft_text)
        self.assertEqual(plan.expert_advice, "保留左侧建筑立面，优先改善右侧停留区")
        self.assertFalse(plan.rag_references)
        self.assertIn(plan.expert_advice, plan.draft_text)
        self.assertTrue(any("保留左侧建筑立面" in item for item in plan.unchanged_regions))
        self.assertTrue(any("优先改善右侧停留区" in item.position for item in plan.object_actions))

    def test_task3_rule_fallback_covers_reverse_metric_directions(self) -> None:
        cartographer = CartographerAgent(knowledge_base=self.kb)
        target_metrics = {
            **self.baseline_metrics,
            "blue_view": 0.02,
            "sky_view": 0.15,
            "built_ratio": 0.58,
            "edge_density": 0.13,
        }
        with (
            patch("agents.cartographer_agent.llm_client.chat", return_value=None),
            patch("agents.cartographer_agent.llm_client.chat_with_image", return_value=None),
        ):
            plan = cartographer.run(
                baseline_metrics=self.baseline_metrics,
                target_metrics=target_metrics,
                scene_context="社区蓝绿空间",
                expert_advice="保持道路结构不变，优先改善入口区域",
                language="zh",
            )

        object_types = "；".join(item.object_type for item in plan.object_actions)
        self.assertIn("蓝色视觉元素", object_types)
        self.assertIn("乔木冠层或通透轻型棚架", object_types)
        self.assertIn("低矮街道家具", object_types)
        self.assertIn("导向性线性元素", object_types)
        self.assertIn("保持全景左右接缝连续", plan.draft_text)
        self.assertIn("优先改善入口区域", plan.draft_text)

    def test_bluegreen_mock_records_run_task2_to_task3_offline(self) -> None:
        payload = json.loads(
            (ROOT / "examples" / "bluegreen_01_03_mock_data.json").read_text(
                encoding="utf-8"
            )
        )
        translator = TranslatorAgent(knowledge_base=self.kb)
        cartographer = CartographerAgent(knowledge_base=self.kb)
        advice = "保持道路与建筑结构不变，优先增加绿化和小尺度可见水体"

        with (
            patch("agents.translator_agent.llm_client.chat", return_value=None),
            patch("agents.translator_agent.llm_client.chat_with_image", return_value=None),
            patch("agents.cartographer_agent.llm_client.chat", return_value=None),
            patch("agents.cartographer_agent.llm_client.chat_with_image", return_value=None),
        ):
            for record in payload["records"]:
                translation = translator.run(
                    experience_baseline=record["experience_baseline"],
                    experience_targets=record["experience_targets"],
                    baseline_metrics=record["baseline_metrics"],
                    scene_context=record["scene_context_text"],
                    original_image_path=record["original_image_path"],
                )
                plan = cartographer.run(
                    baseline_metrics=translation.baseline_metrics.as_dict(),
                    target_metrics=translation.target_metrics.as_dict(),
                    experience_baseline=record["experience_baseline"],
                    experience_targets=record["experience_targets"],
                    scene_context=record["scene_context_text"],
                    original_image_path=record["original_image_path"],
                    expert_advice=advice,
                    language="zh",
                )

                self.assertEqual(len(translation.target_metrics.as_dict()), 7)
                self.assertTrue(translation.conversion_basis)
                self.assertIn(advice, plan.draft_text)
                self.assertTrue(plan.object_actions)

    def test_task3_normalizes_qwen_action_fields_and_appends_hard_constraints(self) -> None:
        response = json.dumps(
            {
                "plan_summary": "模型方案",
                "object_actions": [
                    {
                        "action": "增加",
                        "object_type": "入口绿化",
                        "position": "入口右侧",
                        "quantity": 2,
                        "attributes": "乡土植物，低维护",
                        "rationale": "改善自然感",
                    }
                ],
                "spatial_relations": ["避开入口"],
                "unchanged_regions": ["保留建筑"],
                "constraints": ["保持真实尺度"],
                "modification_text": "在入口右侧增加绿化。",
            },
            ensure_ascii=False,
        )
        with patch(
            "agents.cartographer_agent.llm_client.chat_with_image",
            return_value=response,
        ):
            plan = CartographerAgent(knowledge_base=self.kb).run(
                baseline_metrics=self.baseline_metrics,
                target_metrics={**self.baseline_metrics, "green_view": 0.3},
                original_image_path=str(ROOT / "data" / "p1.jpg"),
                expert_advice="保持道路不变",
                language="zh",
            )

        qwen_action = next(
            item for item in plan.object_actions if item.object_type == "入口绿化"
        )
        self.assertEqual(qwen_action.action, "add")
        self.assertEqual(qwen_action.quantity, "2")
        self.assertEqual(qwen_action.attributes, ["乡土植物", "低维护"])
        self.assertIn("不得擅自删除或移动电线", plan.draft_text)
        self.assertIn("保持全景左右接缝连续", plan.draft_text)


if __name__ == "__main__":
    unittest.main()
