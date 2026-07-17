"""Gradio 前端的纯本地契约测试。"""

from __future__ import annotations

import unittest

from app import gradio_app
from config import MORPH_KEYS


class GradioFrontendTests(unittest.TestCase):
    def test_participants_keep_every_person_and_all_seven_scores(self) -> None:
        rows = [
            ["P01", "甲", 5, 4, 4, 3, 2, 4, 5],
            ["P02", "乙", 3, 3, 4, 4, 1, 3, 4],
        ]
        records = gradio_app.parse_participants(rows)
        self.assertEqual([item["person_id"] for item in records], ["P01", "P02"])
        self.assertEqual(len(records[0]["experience"]), 7)
        self.assertEqual(records[1]["experience"]["environmental_disturbance"], 1.0)

    def test_incomplete_participant_is_rejected_in_chinese(self) -> None:
        with self.assertRaisesRegex(ValueError, "必须填写 1～5 分"):
            gradio_app.parse_participants([["P01", "甲", 3, 3, 3]])

    def test_percentage_sliders_convert_to_internal_zero_to_one(self) -> None:
        values = [25, 5, 40, 70, 12, 18, 4]
        metrics = gradio_app.sliders_to_metrics(values)
        self.assertEqual(list(metrics), MORPH_KEYS)
        self.assertEqual(metrics["green_view"], 0.25)
        self.assertEqual(metrics["color_richness"], 12.0)
        self.assertEqual(metrics["skyline_variance"], 0.04)

    def test_page_builds_without_online_task1_dependency(self) -> None:
        self.assertEqual(type(gradio_app.demo).__name__, "Blocks")
        self.assertGreater(len(gradio_app.demo.blocks), 100)
        source = __import__("inspect").getsource(gradio_app)
        self.assertNotIn("MorphMetricsExtractor", source)
        self.assertNotIn("SESSION =", source)
        self.assertNotIn("人机协同的全景空间优化与经验学习", source)
        self.assertNotIn("gr.File(", source)
        self.assertIn("从后端选择全景图", source)
        self.assertIn("从后端选择项目大表", source)


if __name__ == "__main__":
    unittest.main()
