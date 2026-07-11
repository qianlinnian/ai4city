from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    name: str = "urban-evidence-engine"
    mode: str = "mock"
    random_seed: int = 42


class PathsConfig(BaseModel):
    metadata_csv: Path
    runs_dir: Path
    strict_files: bool = True


class ContextConfig(BaseModel):
    times: list[str] = Field(min_length=1)
    crowd_levels: list[str] = Field(min_length=1)
    noise_db: list[int] = Field(min_length=1)


class ExpertConfig(BaseModel):
    auto_approve: bool = False
    require_reason: bool = True


class VRConfig(BaseModel):
    ethics_approved: bool = False
    exposure_seconds: int = Field(default=60, ge=15, le=180)
    max_scenes_per_participant: int = Field(default=6, ge=1)


class ScoringConfig(BaseModel):
    dimensions: list[str] = Field(min_length=1)


class ValidationConfig(BaseModel):
    alpha: float = Field(default=0.05, gt=0, lt=1)
    minimum_participants: int = Field(default=8, ge=1)
    allow_unvalidated_optimization: bool = False


class AppConfig(BaseModel):
    project: ProjectConfig
    paths: PathsConfig
    context: ContextConfig
    expert: ExpertConfig
    vr: VRConfig
    scoring: ScoringConfig
    validation: ValidationConfig


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)

    config = AppConfig.model_validate(raw)
    base = config_path.parent
    metadata_csv = _resolve(base, config.paths.metadata_csv)
    runs_dir = _resolve(base, config.paths.runs_dir)
    return config.model_copy(
        update={
            "paths": config.paths.model_copy(
                update={"metadata_csv": metadata_csv, "runs_dir": runs_dir}
            )
        }
    )


def _resolve(base: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (base / value).resolve()
