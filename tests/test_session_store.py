from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

from app.session_store import (
    COMPLETED,
    INPUT_PENDING,
    MORPH_REVIEW,
    PLAN_REVIEW,
    SESSION_STAGES,
    InvalidSessionError,
    SessionNotFoundError,
    SessionStore,
    allowed_rollback_stages,
    can_rollback,
    previous_stage,
)


class SessionStoreTestCase(unittest.TestCase):
    def setUp(self) -> None:
        cache_root = Path(__file__).resolve().parents[1] / ".cache" / "test_session_store"
        root = cache_root / uuid.uuid4().hex
        root.mkdir(parents=True)
        self.test_root = root
        self.session_dir = root / "outputs" / "sessions"
        self.upload_dir = root / "uploads"
        self.store = SessionStore(self.session_dir, self.upload_dir)

    def tearDown(self) -> None:
        cache_root = (Path(__file__).resolve().parents[1] / ".cache" / "test_session_store").resolve()
        target = self.test_root.resolve()
        if target.is_relative_to(cache_root):
            shutil.rmtree(target, ignore_errors=True)

    def test_create_generates_unique_ids_and_persists_json(self) -> None:
        first = self.store.create_session({"project_name": "滨水空间"})
        second = self.store.create_session()

        self.assertNotEqual(first["session_id"], second["session_id"])
        self.assertEqual(len(first["session_id"]), 32)
        self.assertEqual(first["stage"], INPUT_PENDING)
        self.assertIn("created_at", first)
        self.assertIn("updated_at", first)

        path = self.session_dir / f"{first['session_id']}.json"
        self.assertTrue(path.is_file())
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["project_name"], "滨水空间")

    def test_save_and_load_session_round_trip_without_global_state(self) -> None:
        state = self.store.create_session({"nested": {"value": 1}})
        state["nested"]["value"] = 2
        state["stage"] = MORPH_REVIEW
        saved = self.store.save_session(state)

        restored = self.store.load_session(saved["session_id"])
        self.assertEqual(restored["nested"], {"value": 2})
        self.assertEqual(restored["stage"], MORPH_REVIEW)
        self.assertFalse(list(self.session_dir.glob("*.tmp")))

    def test_uploads_are_isolated_and_never_overwrite_same_name(self) -> None:
        state = self.store.create_session()
        source = self.test_root / "gradio-temp.bin"
        source.write_bytes(b"panorama")

        first = self.store.save_upload(
            source,
            state["session_id"],
            original_name="同名全景图.jpg",
        )
        second = self.store.save_upload(
            source,
            state["session_id"],
            original_name="同名全景图.jpg",
        )

        self.assertNotEqual(first, second)
        self.assertEqual(first.parent, self.upload_dir / state["session_id"])
        self.assertEqual(first.suffix, ".jpg")
        self.assertEqual(first.read_bytes(), b"panorama")
        self.assertEqual(second.read_bytes(), b"panorama")

    def test_upload_name_cannot_escape_session_directory(self) -> None:
        state = self.store.create_session()
        source = self.test_root / "source.xlsx"
        source.write_bytes(b"excel")

        saved = self.store.save_upload(
            source,
            state["session_id"],
            original_name="../../outside.xlsx",
        )

        self.assertEqual(saved.parent, self.upload_dir / state["session_id"])
        self.assertNotIn("..", saved.name)
        self.assertFalse((self.test_root / "outside.xlsx").exists())

    def test_restore_reports_missing_corrupt_and_unsafe_sessions(self) -> None:
        with self.assertRaisesRegex(InvalidSessionError, "session_id 格式无效"):
            self.store.load_session("../secrets")

        missing = "a" * 32
        with self.assertRaisesRegex(SessionNotFoundError, "未找到会话"):
            self.store.load_session(missing)

        broken = "b" * 32
        (self.session_dir / f"{broken}.json").write_text("{", encoding="utf-8")
        with self.assertRaisesRegex(InvalidSessionError, "会话文件已损坏"):
            self.store.load_session(broken)

    def test_non_json_state_and_invalid_stage_have_clear_errors(self) -> None:
        state = self.store.create_session()
        state["not_json"] = Path("image.jpg")
        with self.assertRaisesRegex(InvalidSessionError, "无法转换为 JSON"):
            self.store.save_session(state)

        state.pop("not_json")
        state["stage"] = "unknown"
        with self.assertRaisesRegex(InvalidSessionError, "未知会话阶段"):
            self.store.save_session(state)

    def test_stage_helpers_and_persisted_rollback(self) -> None:
        self.assertEqual(len(SESSION_STAGES), 9)
        self.assertTrue(can_rollback(COMPLETED, PLAN_REVIEW))
        self.assertFalse(can_rollback(PLAN_REVIEW, COMPLETED))
        self.assertEqual(previous_stage(INPUT_PENDING), None)
        self.assertEqual(previous_stage(MORPH_REVIEW), SESSION_STAGES[1])
        self.assertEqual(
            allowed_rollback_stages(MORPH_REVIEW),
            SESSION_STAGES[:2],
        )

        state = self.store.create_session(stage=COMPLETED)
        rolled_back = self.store.rollback(state, PLAN_REVIEW)
        self.assertEqual(rolled_back["stage"], PLAN_REVIEW)
        self.assertEqual(
            self.store.load_session(state["session_id"])["stage"],
            PLAN_REVIEW,
        )
        with self.assertRaisesRegex(InvalidSessionError, "不能从"):
            self.store.rollback(rolled_back, COMPLETED)


if __name__ == "__main__":
    unittest.main()
