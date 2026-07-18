from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from agents.scene_understanding_agent import SceneUnderstandingAgent
from schemas.models import PanoramaViewMetadata, PanoramaViewSet


ROOT = Path(__file__).resolve().parents[1]


class _FakeViewGenerator:
    def __init__(self, result: PanoramaViewSet) -> None:
        self.result = result

    def generate(self, *_args, **_kwargs) -> PanoramaViewSet:
        return self.result


class SceneUnderstandingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = ROOT / ".cache" / "test_scene_understanding" / uuid.uuid4().hex
        self.root.mkdir(parents=True)
        self.source = self.root / "source.jpg"
        self.view = self.root / "yaw_000.jpg"
        Image.new("RGB", (200, 100), (40, 80, 120)).save(self.source)
        Image.new("RGB", (64, 64), (30, 60, 90)).save(self.view)
        metadata = PanoramaViewMetadata(
            view_id="yaw_000",
            source_image_path=str(self.source),
            source_image_id="scene",
            yaw=0,
            pitch=0,
            fov=90,
            width=64,
            height=64,
            output_path=str(self.view),
        )
        self.view_set = PanoramaViewSet(
            source_image_path=str(self.source),
            source_image_id="scene",
            source_sha256="a" * 64,
            source_width=200,
            source_height=100,
            views=[metadata],
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_schema_filters_unverified_water_and_keeps_evidence(self) -> None:
        response = json.dumps(
            {
                "roads": [
                    {
                        "name": "步行道",
                        "position": "前景",
                        "confidence": 0.9,
                        "evidence_view_ids": ["yaw_000"],
                    }
                ],
                "water": [
                    {
                        "name": "疑似水面",
                        "description": "可能只是蓝色铺装",
                        "confidence": 0.3,
                        "evidence_view_ids": ["yaw_000"],
                    }
                ],
                "buildings": [],
                "entrances": [],
                "vegetation": [],
                "street_furniture": [],
                "infrastructure": [],
                "editable_objects": [],
                "fixed_regions": [],
                "spatial_relations": ["步行道位于建筑前方"],
                "panorama_seam_constraints": ["接缝处保持连续"],
                "ambiguities": [],
                "confidence": 0.8,
                "evidence_view_ids": ["yaw_000"],
            },
            ensure_ascii=False,
        )
        agent = SceneUnderstandingAgent(
            _FakeViewGenerator(self.view_set), enabled=True
        )
        with patch(
            "agents.scene_understanding_agent.llm_client.chat_with_images",
            return_value=response,
        ) as mocked:
            result = agent.run(self.source)

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.roads[0].evidence_view_ids, ["yaw_000"])
        self.assertFalse(result.water)
        self.assertTrue(any("疑似水面" in item for item in result.ambiguities))
        self.assertEqual(len(mocked.call_args.args[2]), 1)

    def test_model_failure_returns_empty_degraded_inventory(self) -> None:
        agent = SceneUnderstandingAgent(
            _FakeViewGenerator(self.view_set), enabled=True
        )
        with patch(
            "agents.scene_understanding_agent.llm_client.chat_with_images",
            return_value=None,
        ):
            result = agent.run(self.source)
        self.assertEqual(result.status, "degraded")
        self.assertFalse(result.roads)
        self.assertFalse(result.water)
        self.assertIn("继续", result.degradation_reason)


if __name__ == "__main__":
    unittest.main()
