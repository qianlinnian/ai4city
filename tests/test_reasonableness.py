from __future__ import annotations

import unittest
from unittest.mock import patch

from agents.cartographer_agent import CartographerAgent
from agents.reasonableness import evaluate_task2_target


class ReasonablenessTests(unittest.TestCase):
    def test_blue_increase_without_verified_water_is_flagged_not_rewritten(self) -> None:
        baseline = {
            "green_view": 0.2,
            "blue_view": 0.05,
            "sky_view": 0.3,
            "built_ratio": 0.5,
            "color_richness": 5.0,
            "edge_density": 0.08,
            "skyline_variance": 0.02,
        }
        target = {**baseline, "blue_view": 0.11}
        report = evaluate_task2_target(
            baseline,
            target,
            {"status": "ok", "water": [], "roads": [], "buildings": []},
        )
        self.assertEqual(report["status"], "warning")
        self.assertTrue(any("未确认真实水体" in item for item in report["warnings"]))
        self.assertEqual(target["blue_view"], 0.11)

        with patch("agents.cartographer_agent.llm_client.chat", return_value=None):
            plan = CartographerAgent().run(
                baseline_metrics=baseline,
                target_metrics=target,
                scene_understanding={
                    "status": "ok",
                    "water": [],
                    "roads": [],
                    "buildings": [],
                    "fixed_regions": [],
                    "infrastructure": [],
                    "panorama_seam_constraints": [],
                },
                language="zh",
            )
        self.assertTrue(any("不得为满足蓝视率" in item for item in plan.constraints))
        self.assertFalse(any("水体" in item.object_type for item in plan.object_actions))


if __name__ == "__main__":
    unittest.main()
