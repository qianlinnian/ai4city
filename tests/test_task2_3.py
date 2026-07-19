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
from knowledge_base.rag_provider import NullRagProvider
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
        translator = TranslatorAgent(
            knowledge_base=self.kb,
            rag_provider=NullRagProvider(),
        )
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
            result = TranslatorAgent(
                knowledge_base=self.kb,
                rag_provider=NullRagProvider(),
            ).run(
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

    def test_task2_revision_prompt_uses_previous_targets_and_knob_changes(self) -> None:
        revised_target = {
            "green_view": 0.34,
            "blue_view": 0.08,
            "sky_view": 0.27,
            "built_ratio": 0.42,
            "color_richness": 7.5,
            "edge_density": 0.065,
            "skyline_variance": 0.026,
        }
        previous_experience = dict(self.experience_targets)
        new_experience = {**previous_experience, "naturalness": 4.8}
        previous_morph = {**self.baseline_metrics, "green_view": 0.30}
        with patch(
            "agents.translator_agent.llm_client.chat_with_image",
            return_value=json.dumps(revised_target),
        ) as mocked_chat:
            result = TranslatorAgent(
                knowledge_base=self.kb,
                rag_provider=NullRagProvider(),
            ).run(
                experience_records=self.experience_records,
                experience_targets=new_experience,
                baseline_metrics=self.baseline_metrics,
                original_image_path=str(ROOT / "data" / "p1.jpg"),
                prompt_variant="revision",
                previous_experience_targets=previous_experience,
                previous_target_metrics=previous_morph,
            )

        system_text, user_text = mocked_chat.call_args.args[:2]
        self.assertIn("修订轮次", system_text)
        self.assertIn('"previous_experience_targets"', user_text)
        self.assertIn('"previous_target_metrics"', user_text)
        self.assertIn('"naturalness": 0.8', user_text)
        self.assertEqual(result.prompt_variant, "revision")
        self.assertIn("修订轮次", result.rationale)

    def test_task3_selects_scene_specific_prompt_profiles(self) -> None:
        scene_cases = [
            ("社区", "community", "社区场景"),
            ("蓝绿", "blue_green", "蓝绿场景"),
            ("商办", "commercial_office", "商办场景"),
        ]
        target = {**self.baseline_metrics, "green_view": 0.32}
        for scene_type, profile_key, profile_label in scene_cases:
            with self.subTest(scene_type=scene_type), patch(
                "agents.cartographer_agent.llm_client.chat",
                return_value=None,
            ) as mocked_chat:
                plan = CartographerAgent(
                    knowledge_base=self.kb,
                    rag_provider=NullRagProvider(),
                ).run(
                    baseline_metrics=self.baseline_metrics,
                    target_metrics=target,
                    scene_type=scene_type,
                    scene_context=f"空间类型: {scene_type}",
                    scene_understanding={"status": "ok"},
                    language="zh",
                )

            system_text, user_text = mocked_chat.call_args.args[:2]
            self.assertEqual(plan.scene_prompt_profile, profile_key)
            self.assertIn(profile_label, system_text)
            self.assertIn("2至5项", system_text)
            self.assertIn(f"场景Prompt模板: {profile_label}", user_text)
            self.assertTrue(any(profile_label in item for item in [plan.plan_summary]))

        with patch("agents.cartographer_agent.llm_client.chat", return_value=None):
            explicit_community = CartographerAgent(
                knowledge_base=self.kb,
                rag_provider=NullRagProvider(),
            ).run(
                baseline_metrics=self.baseline_metrics,
                target_metrics=target,
                scene_type="社区",
                scene_context="住宅邻里紧邻滨水公园",
                language="zh",
            )
        self.assertEqual(explicit_community.scene_prompt_profile, "community")

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
        cartographer = CartographerAgent(
            knowledge_base=self.kb,
            rag_provider=NullRagProvider(),
        )
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
        self.assertIn("保持原始视点", plan.draft_text)
        self.assertIn("【仅执行以下可见修改】", plan.draft_text)
        self.assertIn("【保持不变】", plan.draft_text)
        self.assertIn("【渲染约束】", plan.draft_text)
        self.assertIn("数量/范围（视觉约束，尽量满足）", plan.draft_text)
        self.assertEqual(plan.expert_advice, "保留左侧建筑立面，优先改善右侧停留区")
        self.assertFalse(plan.rag_references)
        self.assertNotIn(plan.expert_advice, plan.draft_text)
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
        self.assertIn("保持全景左右边缘视觉连续", plan.draft_text)
        self.assertIn("优先改善入口区域", plan.draft_text)

    def test_task3_reformats_llm_free_text_as_seedream_sections(self) -> None:
        response = json.dumps(
            {
                "plan_summary": "改善社区入口的遮阴与短暂停留条件",
                "object_actions": [
                    {
                        "action": "add",
                        "object_type": "带靠背木质座椅",
                        "position": "画面右侧现有绿化带内",
                        "quantity": "2组，每组1张",
                        "attributes": ["面向开敞空间", "不占用步行通道"],
                        "rationale": "提升可停留意愿",
                    }
                ],
                "spatial_relations": ["座椅位于树冠遮阴范围内"],
                "unchanged_regions": ["社区入口"],
                "constraints": ["不要新增建筑"],
                "modification_text": "这是一段不应原样堆叠到最终提示词中的自由文本。",
            },
            ensure_ascii=False,
        )
        with patch("agents.cartographer_agent.llm_client.chat", return_value=response):
            plan = CartographerAgent(
                knowledge_base=self.kb,
                rag_provider=NullRagProvider(),
            ).run(
                baseline_metrics=self.baseline_metrics,
                target_metrics={**self.baseline_metrics, "green_view": 0.3},
                scene_type="社区",
                scene_context="社区入口",
                scene_understanding={"status": "ok"},
                language="zh",
            )

        self.assertNotIn("不应原样堆叠", plan.draft_text)
        self.assertIn("位置：画面右侧现有绿化带内", plan.draft_text)
        self.assertIn("2组，每组1张", plan.draft_text)
        self.assertIn("面向开敞空间", plan.draft_text)
        self.assertLess(
            plan.draft_text.index("【仅执行以下可见修改】"),
            plan.draft_text.index("【保持不变】"),
        )

    def test_task3_keeps_workflow_advice_out_of_execution_prompt(self) -> None:
        advice = "前端人工确认形态要素"
        with patch("agents.cartographer_agent.llm_client.chat", return_value=None):
            plan = CartographerAgent(
                knowledge_base=self.kb,
                rag_provider=NullRagProvider(),
            ).run(
                baseline_metrics=self.baseline_metrics,
                target_metrics={**self.baseline_metrics, "green_view": 0.3},
                scene_understanding={"status": "ok"},
                expert_advice=advice,
                language="zh",
            )

        self.assertEqual(plan.expert_advice, advice)
        self.assertNotIn(advice, plan.draft_text)
        self.assertIn("【仅执行以下可见修改】", plan.draft_text)

    def test_task3_removes_unsafe_cable_instruction_from_mixed_action(self) -> None:
        response = json.dumps(
            {
                "plan_summary": "改善树冠通透性",
                "object_actions": [
                    {
                        "action": "adjust",
                        "object_type": "乔木枝叶及空中线缆",
                        "position": "画面中景乔木下层",
                        "quantity": "离地2.5米以上",
                        "attributes": ["修剪下层枝条", "清理杂乱线缆"],
                    }
                ],
                "spatial_relations": [],
                "unchanged_regions": [],
                "constraints": [],
            },
            ensure_ascii=False,
        )
        with patch("agents.cartographer_agent.llm_client.chat", return_value=response):
            plan = CartographerAgent(
                knowledge_base=self.kb,
                rag_provider=NullRagProvider(),
            ).run(
                baseline_metrics=self.baseline_metrics,
                target_metrics={**self.baseline_metrics, "sky_view": 0.5},
                scene_understanding={"status": "ok"},
                language="zh",
            )

        self.assertIn("乔木枝叶", plan.draft_text)
        self.assertNotIn("乔木枝叶及空中线缆", plan.draft_text)
        self.assertNotIn("清理杂乱线缆", plan.draft_text)

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
                self.assertNotIn(advice, plan.draft_text)
                self.assertIn("保持道路与建筑结构不变", plan.draft_text)
                self.assertIn("优先增加绿化和小尺度可见水体", plan.draft_text)
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
        self.assertIn("必要基础设施不变", plan.draft_text)
        self.assertIn("保持全景左右边缘视觉连续", plan.draft_text)

    def test_task3_accepts_degraded_string_scene_elements(self) -> None:
        with patch("agents.cartographer_agent.llm_client.chat", return_value=None):
            plan = CartographerAgent(knowledge_base=self.kb).run(
                baseline_metrics=self.baseline_metrics,
                target_metrics={**self.baseline_metrics, "green_view": 0.3},
                scene_understanding={
                    "status": "ok",
                    "fixed_regions": ["既有建筑与道路"],
                    "infrastructure": ["消防通道"],
                    "panorama_seam_constraints": ["保持接缝连续"],
                },
                language="zh",
            )

        self.assertIn("场景清单固定区域：既有建筑与道路", plan.unchanged_regions)
        self.assertIn("已识别基础设施：消防通道", plan.unchanged_regions)
        self.assertIn("保持接缝连续", plan.constraints)


if __name__ == "__main__":
    unittest.main()
