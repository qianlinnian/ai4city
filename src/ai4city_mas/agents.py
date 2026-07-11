from __future__ import annotations

import csv
from itertools import product
from pathlib import Path
from typing import Any

from ai4city_mas.adapters import AdapterRegistry
from ai4city_mas.config import AppConfig
from ai4city_mas.domain import SceneRecord


class DatasetLoader:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def load(self) -> list[dict[str, Any]]:
        metadata_path = self.config.paths.metadata_csv
        with metadata_path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))

        scenes: list[dict[str, Any]] = []
        for row in rows:
            image_path = _asset_path(metadata_path.parent, row["image_path"])
            video_path = _asset_path(metadata_path.parent, row.get("video_path", ""))
            if self.config.paths.strict_files and not image_path.exists():
                raise FileNotFoundError(f"Missing panorama for {row['scene_id']}: {image_path}")
            if (
                self.config.paths.strict_files
                and video_path is not None
                and not video_path.exists()
            ):
                raise FileNotFoundError(f"Missing video for {row['scene_id']}: {video_path}")

            scene = SceneRecord(
                scene_id=row["scene_id"],
                image_path=str(image_path),
                video_path=str(video_path) if video_path else None,
                space_type=row["space_type"],
                captured_at=row.get("captured_at") or None,
                gps=row.get("gps") or None,
                noise_db=_optional_float(row.get("noise_db")),
                crowd_level=row.get("crowd_level") or None,
                can_enter=_optional_bool(row.get("can_enter")),
                can_sit=_optional_bool(row.get("can_sit")),
            )
            scenes.append(scene.model_dump(mode="json"))
        return scenes


class EAgent:
    def __init__(self, adapters: AdapterRegistry) -> None:
        self.adapters = adapters

    def describe(self, scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            self.adapters.perception.describe(SceneRecord.model_validate(scene))
            for scene in scenes
        ]

    def extract_features(
        self, descriptions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return [self.adapters.features.extract(item) for item in descriptions]


class ExpertGate:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def calibrate(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        approved = self.config.expert.auto_approve
        return {
            "status": "approved" if approved else "awaiting_expert",
            "decision_source": "demo_config" if approved else None,
            "require_reason": self.config.expert.require_reason,
            "reviewed_scene_ids": [item["scene_id"] for item in features],
            "revisions": [],
            "note": (
                "Auto-approved for backbone smoke testing only."
                if approved
                else "Expert must review descriptions, weights, labels, and boundaries."
            ),
        }


class CAgent:
    def __init__(self, config: AppConfig, adapters: AdapterRegistry) -> None:
        self.config = config
        self.adapters = adapters

    def build_context_matrix(
        self, scenes: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        contexts: list[dict[str, Any]] = []
        combinations = product(
            self.config.context.times,
            self.config.context.crowd_levels,
            self.config.context.noise_db,
        )
        templates = list(combinations)
        for scene in scenes:
            for index, (time_name, crowd, noise) in enumerate(templates, start=1):
                contexts.append(
                    {
                        "scene_id": scene["scene_id"],
                        "situation_id": f"{scene['scene_id']}-ctx-{index:02d}",
                        "time": time_name,
                        "crowd_level": crowd,
                        "noise_db": noise,
                    }
                )
        return contexts

    def score(
        self,
        features: list[dict[str, Any]],
        contexts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        feature_by_scene = {item["scene_id"]: item for item in features}
        records: list[dict[str, Any]] = []
        for context in contexts:
            result = self.adapters.scoring.score(
                feature_by_scene[context["scene_id"]],
                context,
                self.config.scoring.dimensions,
            )
            records.append({**context, **result})
        return records


class DAgent:
    def propose_interventions(
        self,
        scenes: list[dict[str, Any]],
        features: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        feature_by_scene = {item["scene_id"]: item for item in features}
        records = []
        for scene in scenes:
            current_gvf = feature_by_scene[scene["scene_id"]]["features"][
                "green_view_factor"
            ]
            records.append(
                {
                    "scene_id": scene["scene_id"],
                    "evidence_level": "hypothesis",
                    "structural_lock": True,
                    "variants": [
                        {
                            "variant_id": "shade-and-seat",
                            "parameters": {"shade": "+", "seating": "+"},
                        },
                        {
                            "variant_id": "vertical-green",
                            "parameters": {
                                "green_view_factor_baseline": current_gvf,
                                "green_view_factor_delta": 0.10,
                            },
                        },
                        {
                            "variant_id": "visibility-and-light",
                            "parameters": {"boundary_openness": "+", "lighting": "+"},
                        },
                    ],
                }
            )
        return records

    def optimize(
        self,
        interventions: list[dict[str, Any]],
        validation: dict[str, Any],
    ) -> dict[str, Any]:
        validated = validation["status"] == "validated"
        return {
            "status": "ready_for_export" if validated else "draft_only",
            "evidence_level": "empirically_validated" if validated else "hypothesis",
            "hard_constraints": validation.get("validated_thresholds", []) if validated else [],
            "warning": None if validated else "No observed VR/physiology data; do not lock thresholds.",
            "strategy_count": len(interventions),
            "export_targets": ["grasshopper_json", "dashboard_api"],
        }


class XAgent:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def build_experiment_plan(
        self, interventions: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            "status": "planned" if self.config.vr.ethics_approved else "awaiting_ethics",
            "ethics_approved": self.config.vr.ethics_approved,
            "exposure_seconds": self.config.vr.exposure_seconds,
            "max_scenes_per_participant": self.config.vr.max_scenes_per_participant,
            "scene_ids": [item["scene_id"] for item in interventions[:3]],
            "measures": [
                "naturalness",
                "comfort",
                "restoration",
                "pleasure",
                "stimulation",
                "safety",
                "stay_intention",
                "hrv_optional",
                "eda_optional",
            ],
            "safety": {"stop_on_discomfort": True, "deidentified_participants": True},
        }


class HAgent:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def validate(self, scores: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "status": "pending_observed_data",
            "alpha": self.config.validation.alpha,
            "minimum_participants": self.config.validation.minimum_participants,
            "predicted_record_count": len(scores),
            "observed_record_count": 0,
            "validated_thresholds": [],
            "analysis_plan": [
                "signal_quality_and_timestamp_alignment",
                "mixed_effects_or_repeated_measures_model",
                "prediction_observation_agreement",
                "nonlinear_threshold_estimation",
                "expert_diagnosis_of_outliers",
            ],
            "warning": "Example thresholds from project documents were not treated as evidence.",
        }


def build_evidence_package(
    project_name: str,
    artifacts: dict[str, dict[str, Any]],
    validation: dict[str, Any],
    optimization: dict[str, Any],
) -> dict[str, Any]:
    risk_flags = []
    if validation["observed_record_count"] == 0:
        risk_flags.append("mock_model_outputs")
    if not validation["validated_thresholds"]:
        risk_flags.append("thresholds_not_validated")
    return {
        "project": project_name,
        "artifact_index": artifacts,
        "validation_status": validation["status"],
        "design_evidence_level": optimization["evidence_level"],
        "risk_flags": risk_flags,
        "dashboard_contract": {
            "pipeline_status": "manifest.json",
            "scenario_analysis": "05_context_matrix + 06_ai_scoring",
            "evidence_logic": "09_human_validation",
            "decision_dashboard": "10_parametric_optimization",
        },
    }


def _asset_path(base: Path, raw: str | None) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _optional_float(value: str | None) -> float | None:
    return float(value) if value not in (None, "") else None


def _optional_bool(value: str | None) -> bool | None:
    if value in (None, ""):
        return None
    return value.strip().lower() in {"1", "true", "yes", "y"}
