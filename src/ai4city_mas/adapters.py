from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol

from ai4city_mas.domain import SceneRecord


class PerceptionAdapter(Protocol):
    model_id: str

    def describe(self, scene: SceneRecord) -> dict[str, Any]: ...


class FeatureAdapter(Protocol):
    model_id: str

    def extract(self, description: dict[str, Any]) -> dict[str, Any]: ...


class ScoringAdapter(Protocol):
    model_id: str

    def score(
        self,
        features: dict[str, Any],
        context: dict[str, Any],
        dimensions: list[str],
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class AdapterRegistry:
    perception: PerceptionAdapter
    features: FeatureAdapter
    scoring: ScoringAdapter


class MockPerceptionAdapter:
    model_id = "mock-perception-v1"

    def describe(self, scene: SceneRecord) -> dict[str, Any]:
        return {
            "scene_id": scene.scene_id,
            "model_id": self.model_id,
            "prompt_version": "ecological-social-spatial-v1",
            "description": {
                "space_type": scene.space_type,
                "ecological": "Demo-only structured observation; replace with a VLM adapter.",
                "social": f"crowd={scene.crowd_level or 'unknown'}, sit={scene.can_sit}",
                "environmental_pressure": f"noise_db={scene.noise_db or 'unknown'}",
                "access": {"can_enter": scene.can_enter, "can_sit": scene.can_sit},
            },
            "confidence": 0.0,
            "evidence_level": "mock",
        }


class MockFeatureAdapter:
    model_id = "mock-feature-v1"

    def extract(self, description: dict[str, Any]) -> dict[str, Any]:
        scene_id = description["scene_id"]
        return {
            "scene_id": scene_id,
            "model_id": self.model_id,
            "features": {
                "green_view_factor": round(0.15 + 0.35 * _unit(scene_id, "gvf"), 4),
                "sky_view_factor": round(0.10 + 0.50 * _unit(scene_id, "svf"), 4),
                "shade_ratio": round(0.20 + 0.55 * _unit(scene_id, "shade"), 4),
                "enclosure": round(0.30 + 0.60 * _unit(scene_id, "enclosure"), 4),
                "interface_monotony": round(_unit(scene_id, "monotony"), 4),
            },
            "evidence_level": "mock",
        }


class MockScoringAdapter:
    model_id = "mock-context-score-v1"

    def score(
        self,
        features: dict[str, Any],
        context: dict[str, Any],
        dimensions: list[str],
    ) -> dict[str, Any]:
        values = features["features"]
        crowd_penalty = {"low": 0.0, "medium": 0.25, "high": 0.55}.get(
            context["crowd_level"], 0.25
        )
        noise_penalty = max(0.0, (float(context["noise_db"]) - 55.0) / 30.0)
        base = (
            2.8
            + values["green_view_factor"] * 1.2
            + values["sky_view_factor"] * 0.5
            + values["shade_ratio"] * 0.3
            - crowd_penalty
            - noise_penalty
        )
        scores = {}
        for dimension in dimensions:
            offset = (_unit(features["scene_id"], dimension) - 0.5) * 0.5
            scores[dimension] = round(min(5.0, max(0.0, base + offset)), 3)
        return {
            "model_id": self.model_id,
            "scores": scores,
            "confidence": 0.0,
            "evidence_level": "mock",
        }


def default_adapters() -> AdapterRegistry:
    return AdapterRegistry(
        perception=MockPerceptionAdapter(),
        features=MockFeatureAdapter(),
        scoring=MockScoringAdapter(),
    )


def _unit(*parts: str) -> float:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)
