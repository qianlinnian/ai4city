from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pipeline.orchestrator import (
    COMPLETED,
    GENERATED,
    INPUT_PENDING,
    MORPH_REVIEW,
    PLAN_CONFIRMED,
    PLAN_REVIEW,
    VALIDATION_PENDING,
    PipelineOrchestrator,
)
from schemas.models import (
    GenerationResult,
    ModificationPlan,
    MorphMetrics,
    MorphTranslationResult,
)


BASELINE = {
    "green_view": 0.20,
    "blue_view": 0.05,
    "sky_view": 0.30,
    "built_ratio": 0.40,
    "color_richness": 12,
    "edge_density": 0.10,
    "skyline_variance": 0.03,
}

EXPERIENCE = {
    "comfort": 4,
    "naturalness": 4,
    "safety": 4,
    "relaxation": 4,
    "environmental_disturbance": 2,
    "stay_intention": 4,
    "overall_impression": 4,
}


class FakeTranslator:
    def run(self, **kwargs):
        target = {**kwargs["baseline_metrics"], "green_view": 0.28}
        return MorphTranslationResult(
            baseline_metrics=MorphMetrics(**kwargs["baseline_metrics"]),
            target_metrics=MorphMetrics(**target),
            experience_baseline=kwargs.get("experience_baseline") or EXPERIENCE,
            experience_records=kwargs.get("experience_records") or [],
            experience_targets=kwargs["experience_targets"],
        )

    def apply_human_override(self, result, human_metrics=None, note=""):
        if not human_metrics:
            return result
        data = result.model_dump()
        data["target_metrics"].update(human_metrics)
        return MorphTranslationResult(**data)


class FakeCartographer:
    def run(self, **kwargs):
        return ModificationPlan(
            draft_text="Add trees without changing the landmark.",
            object_actions=[],
            constraints=["keep landmark"],
            original_image_path=kwargs["original_image_path"],
        )

    def apply_human_edit(self, plan, human_plan):
        data = plan.model_dump()
        data["draft_text"] = human_plan
        return ModificationPlan(**data)


class FakeLearning:
    def record_translation_feedback(self, **kwargs):
        return None


class FakeMemory:
    def __init__(self):
        self.last_payload = None

    def store_feedback(self, **kwargs):
        self.last_payload = kwargs
        return SimpleNamespace(id="memory-test")


class FakeGenerator:
    def run(self, *, image_path, prompt):
        return GenerationResult(
            output_image_path=str(image_path),
            prompt_used=prompt,
            mock=True,
            raw={"backend": "mock"},
        )


class PipelineExcelFlowTests(unittest.TestCase):
    def setUp(self):
        # Reuse this existing file as a path-only fixture.  The pipeline never
        # decodes it, and this keeps the test compatible with read-only runners.
        self.image = Path(__file__)
        self.memory = FakeMemory()
        self.pipe = PipelineOrchestrator(
            generator=FakeGenerator(),
            translator=FakeTranslator(),
            cartographer=FakeCartographer(),
            learning=FakeLearning(),
            memory=self.memory,
        )
        self.session_patch = patch.object(self.pipe, "_persist", return_value=None)
        self.session_patch.start()

    def tearDown(self):
        self.session_patch.stop()

    def _through_generation(self):
        state = self.pipe.start_session(self.image, BASELINE)
        self.assertEqual(state["stage"], INPUT_PENDING)
        self.assertEqual(state["baseline_metrics"], BASELINE)

        state = self.pipe.run_translator(state, EXPERIENCE)
        self.assertEqual(state["stage"], MORPH_REVIEW)
        state = self.pipe.confirm_morph(state)
        self.assertEqual(state["stage"], PLAN_REVIEW)
        state = self.pipe.confirm_plan(state, state["final_prompt"])
        self.assertEqual(state["stage"], PLAN_CONFIRMED)
        state = self.pipe.generate_and_check(state)
        self.assertEqual(state["stage"], GENERATED)
        return state

    def test_pipeline_completes_without_post_edit_metrics(self):
        state = self._through_generation()
        self.assertIsNone(state["quality_report"])
        state = self.pipe.record_post_experience(
            state,
            [
                {
                    "person_id": "p1",
                    "person_name": "participant",
                    "experience": EXPERIENCE,
                }
            ],
        )
        self.assertEqual(state["stage"], VALIDATION_PENDING)
        state = self.pipe.save_memory(state, score=4)
        self.assertEqual(state["stage"], COMPLETED)
        self.assertEqual(self.memory.last_payload["measured_after"], {})

    def test_external_post_edit_metrics_create_quality_report(self):
        state = self._through_generation()
        modified = {**BASELINE, "green_view": 0.27}
        state = self.pipe.record_quality_metrics(state, modified)
        self.assertEqual(state["stage"], VALIDATION_PENDING)
        self.assertEqual(state["post_edit_metrics"], modified)
        self.assertIn("green_view", state["quality_report"]["deviations"])

    def test_baseline_requires_all_seven_valid_values(self):
        incomplete = dict(BASELINE)
        incomplete.pop("sky_view")
        with self.assertRaisesRegex(ValueError, "缺少指标"):
            self.pipe.start_session(self.image, incomplete)

        invalid = {**BASELINE, "color_richness": 25}
        with self.assertRaisesRegex(ValueError, "超出允许范围"):
            self.pipe.start_session(self.image, invalid)

    def test_stage_validation_prevents_skipping(self):
        state = self.pipe.start_session(self.image, BASELINE)
        with self.assertRaisesRegex(ValueError, "当前阶段"):
            self.pipe.confirm_plan(state, "skip")


if __name__ == "__main__":
    unittest.main()
