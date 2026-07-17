from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.generation_backend import default_backend, generate


def _source_image(tmp_path: Path) -> Path:
    source = tmp_path / "panorama.jpg"
    # Mock generation copies bytes and intentionally does not decode images.
    source.write_bytes(b"local-test-image")
    return source


class GenerationBackendTests(unittest.TestCase):
    def setUp(self):
        # Keep test artifacts inside the writable repository workspace.  Some
        # managed Windows environments intentionally restrict the global temp
        # directory even when ``tempfile`` can create a directory there.
        self.temp_dir = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_generate_mock_returns_local_generation_result(self):
        source = _source_image(self.tmp_path)
        target_dir = self.tmp_path / "generated"
        import config

        with patch.object(config, "TARGET_IMG_DIR", target_dir):
            result = generate(source, "  add more trees  ", backend="mock")

        output = Path(result.output_image_path)
        self.assertTrue(result.mock)
        self.assertEqual(result.prompt_used, "add more trees")
        self.assertEqual(result.raw, {"backend": "mock", "status": "mock"})
        self.assertEqual(output.parent, target_dir)
        self.assertEqual(output.read_bytes(), source.read_bytes())
        self.assertNotEqual(output, source)

    def test_generate_uses_environment_default_mock(self):
        source = _source_image(self.tmp_path)
        import config

        clean_env = {"GENERATION_BACKEND": "mock"}
        with (
            patch.object(config, "TARGET_IMG_DIR", self.tmp_path / "generated"),
            patch.dict(os.environ, clean_env, clear=False),
            patch.dict(os.environ, {"IMAGE_GENERATION_BACKEND": ""}, clear=False),
        ):
            self.assertEqual(default_backend(), "mock")
            self.assertTrue(generate(source, "preserve the skyline").mock)

    def test_generate_rejects_unknown_backend(self):
        source = _source_image(self.tmp_path)
        for backend in ("unknown", "openai"):
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(ValueError, "未知生成后端"):
                    generate(source, "valid prompt", backend=backend)

    def test_generate_validates_input_before_loading_backend(self):
        with self.assertRaisesRegex(FileNotFoundError, "找不到输入图片"):
            generate(self.tmp_path / "missing.jpg", "valid prompt", backend="mock")

        source = _source_image(self.tmp_path)
        with self.assertRaisesRegex(ValueError, "提示词不能为空"):
            generate(source, "   ", backend="mock")


if __name__ == "__main__":
    unittest.main()
