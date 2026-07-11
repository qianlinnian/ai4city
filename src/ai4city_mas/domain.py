from __future__ import annotations

from enum import Enum
from typing import Any, TypedDict

from pydantic import BaseModel


SCHEMA_VERSION = "1.0"


class PipelineStage(str, Enum):
    DATA_INGESTION = "01_data_ingestion"
    PERCEPTION = "02_perception_description"
    FEATURES = "03_feature_extraction"
    EXPERT_CALIBRATION = "04_expert_calibration"
    CONTEXT_MATRIX = "05_context_matrix"
    AI_SCORING = "06_ai_scoring"
    INTERVENTIONS = "07_intervention_stimuli"
    VR_EXPERIMENT = "08_vr_experiment"
    HUMAN_VALIDATION = "09_human_validation"
    PARAMETRIC_OPTIMIZATION = "10_parametric_optimization"
    EVIDENCE_PACKAGE = "11_evidence_package"


STAGE_TITLES = {
    PipelineStage.DATA_INGESTION: "Data ingestion and validation",
    PipelineStage.PERCEPTION: "E-Agent perception description",
    PipelineStage.FEATURES: "E-Agent feature extraction",
    PipelineStage.EXPERT_CALIBRATION: "Expert calibration gate",
    PipelineStage.CONTEXT_MATRIX: "C-Agent context matrix",
    PipelineStage.AI_SCORING: "C-Agent perception scoring",
    PipelineStage.INTERVENTIONS: "D-Agent intervention stimuli",
    PipelineStage.VR_EXPERIMENT: "X-Agent VR experiment",
    PipelineStage.HUMAN_VALIDATION: "H-Agent human-factor validation",
    PipelineStage.PARAMETRIC_OPTIMIZATION: "D-Agent parametric optimization",
    PipelineStage.EVIDENCE_PACKAGE: "Evidence package and reporting",
}


class SceneRecord(BaseModel):
    scene_id: str
    image_path: str
    video_path: str | None = None
    space_type: str
    captured_at: str | None = None
    gps: str | None = None
    noise_db: float | None = None
    crowd_level: str | None = None
    can_enter: bool | None = None
    can_sit: bool | None = None


class ArtifactRef(BaseModel):
    stage: PipelineStage
    path: str
    record_count: int
    schema_version: str = SCHEMA_VERSION


class PipelineState(TypedDict):
    run_id: str
    run_dir: str
    stage: str
    status: str
    artifacts: dict[str, dict[str, Any]]
    messages: list[str]
