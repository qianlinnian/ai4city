"""Gradio 前端的纯本地契约测试。"""

from __future__ import annotations

import unittest
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
        records = gradio_app._df_to_persons(gradio_app._persons_to_df(persons))
        self.assertEqual([item["person_id"] for item in records], ["p1", "p2"])
        self.assertEqual(len(records[0]["experience"]), 7)
        self.assertEqual(records[1]["experience"]["environmental_disturbance"], 1)

    def test_person_experience_rejects_blank_or_out_of_range_scores(self) -> None:
        frame = gradio_app._empty_experience_df(1)
        frame.loc[0, gradio_app.EXPERIENCE_LABELS_ZH["comfort"]] = 6
        with self.assertRaisesRegex(ValueError, "必须位于1到5之间"):
            gradio_app._df_to_persons(frame)

        # pandas 3.x does not allow assigning a string into an int64 column.
        # Use an object column so the UI's empty-cell validation is exercised.
        comfort_col = gradio_app.EXPERIENCE_LABELS_ZH["comfort"]
        frame[comfort_col] = frame[comfort_col].astype(object)
        frame.loc[0, comfort_col] = ""
        with self.assertRaisesRegex(ValueError, "不能为空"):
            gradio_app._df_to_persons(frame)

    def test_percentage_sliders_convert_to_internal_zero_to_one(self) -> None:
        values = [25, 5, 40, 70, 12, 18, 4]
        metrics = gradio_app._sliders_to_metrics(values)
        self.assertEqual(list(metrics), MORPH_KEYS)
        self.assertEqual(metrics["green_view"], 0.25)
        self.assertEqual(metrics["color_richness"], 12.0)
        self.assertEqual(metrics["skyline_variance"], 0.04)

    def test_experience_summary_treats_disturbance_as_reverse_metric(self) -> None:
        baseline = {key: 3.0 for key in gradio_app.EXPERIENCE_KEYS}
        targets = {key: 4.0 for key in gradio_app.EXPERIENCE_KEYS}
        targets["environmental_disturbance"] = 2.0
        post = {key: 4.2 for key in gradio_app.EXPERIENCE_KEYS}
        post["environmental_disturbance"] = 1.5

        frame = gradio_app._build_experience_diff_df(baseline, targets, post)
        disturbance = frame.loc[
            frame["体验指标"]
            == gradio_app.EXPERIENCE_LABELS_ZH["environmental_disturbance"]
        ].iloc[0]
        self.assertEqual(disturbance["目标达成"], "超额达成")

    def test_page_builds_with_post_edit_metrics_adapter(self) -> None:
        demo = gradio_app.build_ui()
        self.assertEqual(type(demo).__name__, "Blocks")
        self.assertGreater(len(demo.blocks), 50)
        source = __import__("inspect").getsource(gradio_app)
        self.assertNotIn("MorphMetricsExtractor", source)
        self.assertNotIn("gr.File(", source)
        self.assertIn("list_scene_choices", source)
        self.assertIn("PipelineOrchestrator(", source)
        self.assertIn("post_edit_metrics_extractor=", source)
        self.assertIn("extract_post_edit_metrics", source)
        self.assertNotIn("app.generation_backend", source)
        self.assertNotIn("SCENE_UNDERSTANDING_ENABLED", source)


if __name__ == "__main__":
    unittest.main()
