from __future__ import annotations

import json
import unittest
from pathlib import Path

from ai4city_mas.config import load_config
from ai4city_mas.domain import PipelineStage
from ai4city_mas.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
TEST_RUN_ROOT = ROOT / "runs"


class PipelineTests(unittest.TestCase):
    def test_mock_pipeline_runs_all_stages_without_inventing_validation(self) -> None:
        config = self._test_config()
        state = Pipeline(config).run(run_id=".test-full")

        self.assertEqual(state["status"], "completed")
        self.assertEqual(len(state["artifacts"]), 11)
        run_dir = Path(state["run_dir"])
        validation = self._data(run_dir, PipelineStage.HUMAN_VALIDATION)
        optimization = self._data(run_dir, PipelineStage.PARAMETRIC_OPTIMIZATION)
        contexts = self._data(run_dir, PipelineStage.CONTEXT_MATRIX)

        self.assertEqual(len(contexts), 36)
        self.assertEqual(validation["status"], "pending_observed_data")
        self.assertEqual(validation["validated_thresholds"], [])
        self.assertEqual(optimization["evidence_level"], "hypothesis")
        self.assertEqual(optimization["hard_constraints"], [])
        with (run_dir / "manifest.json").open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        self.assertTrue(
            all(stage["status"] == "completed" for stage in manifest["stages"].values())
        )

    def test_expert_gate_stops_pipeline_when_not_approved(self) -> None:
        config = self._test_config()
        config = config.model_copy(
            update={"expert": config.expert.model_copy(update={"auto_approve": False})}
        )
        state = Pipeline(config).run(run_id=".test-gate")

        self.assertEqual(state["status"], "awaiting_expert")
        self.assertEqual(len(state["artifacts"]), 4)
        self.assertNotIn(PipelineStage.CONTEXT_MATRIX.value, state["artifacts"])

    def test_vr_ethics_gate_stops_before_human_data_processing(self) -> None:
        config = self._test_config()
        config = config.model_copy(
            update={"vr": config.vr.model_copy(update={"ethics_approved": False})}
        )
        state = Pipeline(config).run(run_id=".test-ethics")

        self.assertEqual(state["status"], "awaiting_ethics")
        self.assertEqual(len(state["artifacts"]), 8)
        self.assertNotIn(PipelineStage.HUMAN_VALIDATION.value, state["artifacts"])

    @staticmethod
    def _test_config():
        config = load_config(ROOT / "configs" / "default.yaml")
        return config.model_copy(
            update={
                "paths": config.paths.model_copy(update={"runs_dir": TEST_RUN_ROOT})
            }
        )

    @staticmethod
    def _data(run_dir: Path, stage: PipelineStage):
        path = run_dir / "artifacts" / f"{stage.value}.json"
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)["data"]


if __name__ == "__main__":
    unittest.main()
