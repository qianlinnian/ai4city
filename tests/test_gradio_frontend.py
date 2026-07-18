"""Gradio 前端的纯本地契约测试。"""

from __future__ import annotations

import unittest
import json

from app import gradio_app
from config import MORPH_KEYS


class GradioFrontendTests(unittest.TestCase):
    def test_participants_keep_every_person_and_all_seven_scores(self) -> None:
        persons = [
            {
                "person_id": "P01",
                "person_name": "甲",
                "experience": {key: 3 for key in gradio_app.EXPERIENCE_KEYS},
            },
            {
                "person_id": "P02",
                "person_name": "乙",
                "experience": {
                    **{key: 4 for key in gradio_app.EXPERIENCE_KEYS},
                    "environmental_disturbance": 1,
                },
            },
        ]
        records = gradio_app._parse_person_experience(
            json.dumps(persons, ensure_ascii=False)
        )
        self.assertEqual([item["person_id"] for item in records], ["P01", "P02"])
        self.assertEqual(len(records[0]["experience"]), 7)
        self.assertEqual(records[1]["experience"]["environmental_disturbance"], 1)

    def test_person_experience_requires_json_array(self) -> None:
        with self.assertRaisesRegex(ValueError, "JSON 数组"):
            gradio_app._parse_person_experience('{"person_id": "P01"}')

    def test_percentage_sliders_convert_to_internal_zero_to_one(self) -> None:
        values = [25, 5, 40, 70, 12, 18, 4]
        metrics = gradio_app._sliders_to_metrics(values)
        self.assertEqual(list(metrics), MORPH_KEYS)
        self.assertEqual(metrics["green_view"], 0.25)
        self.assertEqual(metrics["color_richness"], 12.0)
        self.assertEqual(metrics["skyline_variance"], 0.04)

    def test_page_builds_without_online_task1_dependency(self) -> None:
        demo = gradio_app.build_ui()
        self.assertEqual(type(demo).__name__, "Blocks")
        self.assertGreater(len(demo.blocks), 50)
        source = __import__("inspect").getsource(gradio_app)
        self.assertNotIn("MorphMetricsExtractor", source)
        self.assertNotIn("gr.File(", source)
        self.assertIn("list_scene_choices", source)
        self.assertIn("PipelineOrchestrator(force_metrics_fallback=True)", source)
        self.assertNotIn("app.generation_backend", source)
        self.assertNotIn("SCENE_UNDERSTANDING_ENABLED", source)


if __name__ == "__main__":
    unittest.main()
