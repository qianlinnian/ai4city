"""Gradio 会话的文件持久化与上传文件管理。

该模块不保存任何进程级会话变量。调用方应把当前状态放入 ``gr.State``，
并在每个阶段完成后调用 :class:`SessionStore` 写入磁盘。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


INPUT_PENDING = "input_pending"
EXPERIENCE_CONFIRMED = "experience_confirmed"
MORPH_REVIEW = "morph_review"
MORPH_CONFIRMED = "morph_confirmed"
PLAN_REVIEW = "plan_review"
PLAN_CONFIRMED = "plan_confirmed"
GENERATED = "generated"
VALIDATION_PENDING = "validation_pending"
COMPLETED = "completed"

SESSION_STAGES: tuple[str, ...] = (
    INPUT_PENDING,
    EXPERIENCE_CONFIRMED,
    MORPH_REVIEW,
    MORPH_CONFIRMED,
    PLAN_REVIEW,
    PLAN_CONFIRMED,
    GENERATED,
    VALIDATION_PENDING,
    COMPLETED,
)

_STAGE_INDEX = {stage: index for index, stage in enumerate(SESSION_STAGES)}
_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{7,127}$")
_UNSAFE_FILENAME_PATTERN = re.compile(r"[<>:\"/\\|?*\x00-\x1f]")


class SessionStoreError(RuntimeError):
    """会话读写失败，消息可直接展示给前端用户。"""


class SessionNotFoundError(SessionStoreError):
    """找不到指定会话。"""


class InvalidSessionError(SessionStoreError):
    """会话 ID、阶段或状态内容无效。"""


def new_session_id() -> str:
    """返回不包含路径字符的 32 位随机会话 ID。"""

    return uuid.uuid4().hex


def validate_stage(stage: str) -> str:
    """校验并返回阶段名。"""

    if stage not in _STAGE_INDEX:
        allowed = "、".join(SESSION_STAGES)
        raise InvalidSessionError(f"未知会话阶段 {stage!r}；允许值为：{allowed}")
    return stage


def allowed_rollback_stages(current_stage: str) -> tuple[str, ...]:
    """返回当前阶段可以回退到的所有较早阶段。"""

    current_stage = validate_stage(current_stage)
    return SESSION_STAGES[: _STAGE_INDEX[current_stage]]


def can_rollback(current_stage: str, target_stage: str) -> bool:
    """判断是否可以从当前阶段回退到目标阶段。"""

    current_stage = validate_stage(current_stage)
    target_stage = validate_stage(target_stage)
    return _STAGE_INDEX[target_stage] < _STAGE_INDEX[current_stage]


def previous_stage(current_stage: str) -> str | None:
    """返回紧邻的上一阶段；初始阶段没有上一阶段。"""

    current_stage = validate_stage(current_stage)
    index = _STAGE_INDEX[current_stage]
    return SESSION_STAGES[index - 1] if index else None


class SessionStore:
    """基于 JSON 文件的会话仓库。

    Parameters
    ----------
    session_dir:
        会话 JSON 目录，生产环境通常为 ``outputs/sessions``。
    upload_dir:
        上传文件目录。每个会话拥有独立子目录。
    """

    def __init__(self, session_dir: str | Path, upload_dir: str | Path) -> None:
        self.session_dir = Path(session_dir).resolve()
        self.upload_dir = Path(upload_dir).resolve()
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def create_session(
        self,
        initial_state: Mapping[str, Any] | None = None,
        *,
        stage: str = INPUT_PENDING,
    ) -> dict[str, Any]:
        """创建、持久化并返回一个新会话状态。"""

        validate_stage(stage)
        state = dict(initial_state or {})
        if "session_id" in state:
            raise InvalidSessionError("新建会话时不能预先指定 session_id")
        if "stage" in state:
            raise InvalidSessionError("请通过 stage 参数指定新会话阶段")

        for _ in range(10):
            session_id = new_session_id()
            if not self._session_path(session_id).exists():
                break
        else:  # pragma: no cover - UUID 连续碰撞仅作为防御性分支
            raise SessionStoreError("无法生成唯一 session_id，请重试")

        now = _utc_now()
        state.update(
            {
                "session_id": session_id,
                "stage": stage,
                "created_at": now,
                "updated_at": now,
            }
        )
        return self.save_session(state)

    def save_session(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """原子保存完整状态，并返回实际写入状态的副本。"""

        saved = dict(state)
        session_id = self._validate_session_id(saved.get("session_id"))
        validate_stage(saved.get("stage"))
        saved.setdefault("created_at", _utc_now())
        saved["updated_at"] = _utc_now()

        try:
            payload = json.dumps(saved, ensure_ascii=False, indent=2)
        except (TypeError, ValueError) as exc:
            raise InvalidSessionError(f"会话状态无法转换为 JSON：{exc}") from exc

        destination = self._session_path(session_id)
        self._atomic_write_text(destination, payload)
        return saved

    def load_session(self, session_id: str) -> dict[str, Any]:
        """按 session_id 恢复并校验会话。"""

        session_id = self._validate_session_id(session_id)
        path = self._session_path(session_id)
        if not path.is_file():
            raise SessionNotFoundError(f"未找到会话：{session_id}")

        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise InvalidSessionError(f"会话文件已损坏，无法解析：{session_id}") from exc
        except OSError as exc:
            raise SessionStoreError(f"读取会话失败：{exc}") from exc

        if not isinstance(state, dict):
            raise InvalidSessionError(f"会话文件格式错误：{session_id}")
        if state.get("session_id") != session_id:
            raise InvalidSessionError(f"会话文件中的 session_id 不匹配：{session_id}")
        validate_stage(state.get("stage"))
        return state

    def save_upload(
        self,
        source: str | Path,
        session_id: str,
        *,
        original_name: str | None = None,
    ) -> Path:
        """把上传文件唯一保存到该会话目录，并返回绝对路径。

        ``original_name`` 可用于保留浏览器侧的原始文件名；无论原名是否相同，
        保存名都会带独立 UUID，不会覆盖已有上传。
        """

        session_id = self._validate_session_id(session_id)
        source_path = Path(source)
        if not source_path.is_file():
            raise SessionStoreError(f"上传源文件不存在：{source_path}")

        safe_name = self._safe_filename(original_name or source_path.name)
        unique_name = f"{Path(safe_name).stem}_{uuid.uuid4().hex}{Path(safe_name).suffix}"
        session_upload_dir = (self.upload_dir / session_id).resolve()
        self._ensure_within(session_upload_dir, self.upload_dir, "上传目录")
        session_upload_dir.mkdir(parents=True, exist_ok=True)
        destination = (session_upload_dir / unique_name).resolve()
        self._ensure_within(destination, session_upload_dir, "上传文件")

        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{unique_name}.",
                suffix=".tmp",
                dir=session_upload_dir,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                with source_path.open("rb") as source_handle:
                    shutil.copyfileobj(source_handle, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except OSError as exc:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise SessionStoreError(f"保存上传文件失败：{exc}") from exc

        return destination

    def rollback(self, state: Mapping[str, Any], target_stage: str) -> dict[str, Any]:
        """校验回退方向、更新阶段并原子持久化。"""

        current_stage = state.get("stage")
        if not can_rollback(current_stage, target_stage):
            raise InvalidSessionError(
                f"不能从 {current_stage!r} 回退到 {target_stage!r}"
            )
        rolled_back = dict(state)
        rolled_back["stage"] = target_stage
        return self.save_session(rolled_back)

    @staticmethod
    def _validate_session_id(session_id: Any) -> str:
        if not isinstance(session_id, str) or not _SESSION_ID_PATTERN.fullmatch(session_id):
            raise InvalidSessionError("session_id 格式无效")
        return session_id

    def _session_path(self, session_id: str) -> Path:
        path = (self.session_dir / f"{session_id}.json").resolve()
        self._ensure_within(path, self.session_dir, "会话文件")
        return path

    @staticmethod
    def _safe_filename(filename: str) -> str:
        filename = Path(str(filename)).name
        filename = _UNSAFE_FILENAME_PATTERN.sub("_", filename).strip(" .")
        if not filename or filename in {".", ".."}:
            return "upload"
        return filename[:180]

    @staticmethod
    def _ensure_within(path: Path, parent: Path, label: str) -> None:
        try:
            path.relative_to(parent)
        except ValueError as exc:
            raise InvalidSessionError(f"{label}超出允许目录") from exc

    @staticmethod
    def _atomic_write_text(destination: Path, payload: str) -> None:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except OSError as exc:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise SessionStoreError(f"保存会话失败：{exc}") from exc


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "COMPLETED",
    "EXPERIENCE_CONFIRMED",
    "GENERATED",
    "INPUT_PENDING",
    "InvalidSessionError",
    "MORPH_CONFIRMED",
    "MORPH_REVIEW",
    "PLAN_CONFIRMED",
    "PLAN_REVIEW",
    "SESSION_STAGES",
    "SessionNotFoundError",
    "SessionStore",
    "SessionStoreError",
    "VALIDATION_PENDING",
    "allowed_rollback_stages",
    "can_rollback",
    "new_session_id",
    "previous_stage",
    "validate_stage",
]
