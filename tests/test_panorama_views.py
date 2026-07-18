from __future__ import annotations

import hashlib
import shutil
import unittest
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from utils.panorama_views import (
    PanoramaProjectionConfig,
    PanoramaViewGenerator,
    equirectangular_to_perspective,
)


ROOT = Path(__file__).resolve().parents[1]


class PanoramaViewsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = ROOT / ".cache" / "test_panorama_views" / uuid.uuid4().hex
        self.source_dir = self.root / "source"
        self.output_dir = self.root / "output"
        self.source_dir.mkdir(parents=True)

    def tearDown(self) -> None:
        cache_root = (ROOT / ".cache" / "test_panorama_views").resolve()
        target = self.root.resolve()
        if target.is_relative_to(cache_root):
            shutil.rmtree(target, ignore_errors=True)

    def test_projection_centres_follow_yaw_directions(self) -> None:
        width, height = 360, 180
        panorama = np.tile(np.arange(width, dtype=np.float32), (height, 1))
        expected = {0: 180, 90: 270, 180: 0, 270: 90}
        for yaw, source_x in expected.items():
            view = equirectangular_to_perspective(
                panorama,
                yaw=yaw,
                pitch=0,
                fov=60,
                width=101,
                height=101,
            )
            centre = float(view[50, 50])
            if yaw == 180 and centre > 359:
                centre = 0
            self.assertAlmostEqual(centre, source_x, delta=1.1)

    def test_projection_wraps_the_left_right_seam(self) -> None:
        width, height = 360, 180
        circular_distance = np.minimum(
            np.arange(width, dtype=np.float32),
            width - np.arange(width, dtype=np.float32),
        )
        panorama = np.tile(circular_distance, (height, 1))
        view = equirectangular_to_perspective(
            panorama,
            yaw=180,
            pitch=0,
            fov=40,
            width=101,
            height=51,
        )
        row = view[25]
        self.assertLess(float(row[50]), 1.1)
        self.assertAlmostEqual(float(row[49]), float(row[51]), delta=1.2)
        self.assertLess(float(np.max(np.abs(np.diff(row)))), 3.0)

    def test_metadata_dimensions_source_unchanged_and_cache_reuse(self) -> None:
        source = self.source_dir / "测试全景.png"
        pixels = np.zeros((200, 400, 3), dtype=np.uint8)
        pixels[:, :200] = (30, 120, 220)
        pixels[:, 200:] = (220, 80, 30)
        Image.fromarray(pixels).save(source)
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        config = PanoramaProjectionConfig(
            overview_width=200,
            overview_height=100,
            perspective_width=96,
            perspective_height=96,
            fov=90,
            horizontal_yaws=(0, 90, 180, 270),
        )
        generator = PanoramaViewGenerator(
            self.output_dir,
            config,
            read_only_data_dir=self.source_dir,
        )
        first = generator.generate(source, source_image_id="scene-01")
        second = generator.generate(source, source_image_id="scene-01")

        self.assertEqual(len(first.views), 5)
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(before, hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual((first.source_width, first.source_height), (400, 200))
        for view in first.views:
            self.assertTrue(Path(view.output_path).is_file())
            with Image.open(view.output_path) as image:
                self.assertEqual(image.size, (view.width, view.height))

    def test_non_two_to_one_input_is_rejected(self) -> None:
        source = self.source_dir / "not-equirectangular.png"
        Image.new("RGB", (300, 200)).save(source)
        generator = PanoramaViewGenerator(
            self.output_dir,
            PanoramaProjectionConfig(
                overview_width=100,
                overview_height=50,
                perspective_width=32,
                perspective_height=32,
                strict_aspect=True,
            ),
            read_only_data_dir=self.source_dir,
        )
        with self.assertRaisesRegex(ValueError, "期望 2:1"):
            generator.generate(source)

    def test_downward_views_are_configuration_controlled(self) -> None:
        source = self.source_dir / "downward.png"
        Image.new("RGB", (200, 100), (80, 100, 120)).save(source)
        generator = PanoramaViewGenerator(
            self.output_dir,
            PanoramaProjectionConfig(
                overview_width=100,
                overview_height=50,
                perspective_width=32,
                perspective_height=32,
                horizontal_yaws=(0, 90, 180, 270),
                include_downward=True,
                downward_pitch=-20,
            ),
            read_only_data_dir=self.source_dir,
        )
        result = generator.generate(source)
        downward = [view for view in result.views if view.view_id.startswith("down_")]
        self.assertEqual(len(result.views), 9)
        self.assertEqual({view.pitch for view in downward}, {-20.0})


if __name__ == "__main__":
    unittest.main()
