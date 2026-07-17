from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from app.data_catalog import discover_images, discover_metric_tables


HEADERS = [
    "图像名称",
    "绿视率",
    "蓝视率",
    "天空可视率",
    "人造物占比",
    "色彩丰富度",
    "边缘密度",
    "天际线变化率",
]


class DataCatalogTests(unittest.TestCase):
    def test_scene_manifest_drives_backend_image_dropdown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "p1.jpg").write_bytes(b"image")
            with (root / "scenes.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["scene_id", "image_path", "space_type"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "scene_id": "S01",
                        "image_path": "p1.jpg",
                        "space_type": "community",
                    }
                )
            items = discover_images(root, root / "scenes.csv")
            self.assertEqual(len(items), 1)
            self.assertIn("S01", items[0].label)
            self.assertEqual(Path(items[0].path), (root / "p1.jpg").resolve())

    def test_only_valid_seven_metric_tables_are_listed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "project_metrics.csv"
            with valid.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(HEADERS)
                writer.writerow(["p1.jpg", 0.2, 0.03, 0.25, 0.5, 12, 0.1, 0.02])
            (root / "scenes.csv").write_text(
                "scene_id,image_path\nS01,p1.jpg\n", encoding="utf-8"
            )
            items = discover_metric_tables(root, explicit_path=None)
            self.assertEqual([item.label for item in items], ["project_metrics.csv"])


if __name__ == "__main__":
    unittest.main()
