from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai4city_mas.domain import ArtifactRef, PipelineStage, SCHEMA_VERSION


class ArtifactStore:
    def __init__(self, run_dir: Path, run_id: str) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self.artifact_dir = run_dir / "artifacts"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = run_dir / "manifest.json"
        self._manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "status": "running",
            "created_at": _now(),
            "updated_at": _now(),
            "stages": {},
        }
        self._flush_manifest()

    def write(self, stage: PipelineStage, data: Any) -> ArtifactRef:
        path = self.artifact_dir / f"{stage.value}.json"
        envelope = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "stage": stage.value,
            "created_at": _now(),
            "data": data,
        }
        _write_json(path, envelope)
        return ArtifactRef(
            stage=stage,
            path=str(path.resolve()),
            record_count=_record_count(data),
        )

    def read(self, stage: PipelineStage) -> Any:
        path = self.artifact_dir / f"{stage.value}.json"
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)["data"]

    def record(
        self,
        ref: ArtifactRef,
        stage_status: str,
        pipeline_status: str,
    ) -> None:
        self._manifest["stages"][ref.stage.value] = {
            "status": stage_status,
            "artifact": ref.model_dump(mode="json"),
        }
        self._manifest["status"] = pipeline_status
        self._manifest["updated_at"] = _now()
        self._flush_manifest()

    def finalize(self, status: str, messages: list[str]) -> None:
        self._manifest["status"] = status
        self._manifest["messages"] = messages
        self._manifest["updated_at"] = _now()
        self._flush_manifest()

    def _flush_manifest(self) -> None:
        _write_json(self.manifest_path, self._manifest)


def _write_json(path: Path, payload: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temp.replace(path)


def _record_count(data: Any) -> int:
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        for key in ("records", "scenes", "scores", "contexts", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return len(value)
    return 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
