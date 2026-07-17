"""
数据模型（全流程共用）
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from config import LEGACY_EXPERIENCE_ALIASES


class ExperienceTargets(BaseModel):
    """单人七项 VR 体验感受指标（1-5）。"""

    model_config = ConfigDict(extra="forbid")

    comfort: float = Field(..., ge=1, le=5, description="舒适度")
    naturalness: float = Field(..., ge=1, le=5, description="自然感")
    safety: float = Field(..., ge=1, le=5, description="安全感")
    relaxation: float = Field(..., ge=1, le=5, description="放松感")
    environmental_disturbance: float = Field(
        ...,
        ge=1,
        le=5,
        description="环境干扰感，反向指标，分值越低越好",
    )
    stay_intention: float = Field(..., ge=1, le=5, description="可停留意愿")
    overall_impression: float = Field(..., ge=1, le=5, description="总体感")

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            normalized = dict(data)
            for old_key, new_key in LEGACY_EXPERIENCE_ALIASES.items():
                if new_key not in normalized and old_key in normalized:
                    normalized[new_key] = normalized[old_key]
                normalized.pop(old_key, None)
            return normalized
        return data

    def as_dict(self) -> dict[str, float]:
        return self.model_dump()


class PersonExperience(BaseModel):
    """多人体验：单个人的体验指标"""

    person_id: str
    person_name: str = ""
    experience: ExperienceTargets

    def experience_dict(self) -> dict[str, float]:
        return self.experience.as_dict()


class MultiPersonExperience(BaseModel):
    """同一图像的多人体验指标集合；始终保留逐人记录。"""

    persons: list[PersonExperience] = Field(default_factory=list)

    def as_prompt_records(self) -> list[dict[str, Any]]:
        """返回逐人记录，不求平均、不丢弃个体差异。"""
        return [person.model_dump() for person in self.persons]


class SceneContext(BaseModel):
    """七项情景要素（与全景图配套）。"""

    model_config = ConfigDict(extra="ignore")

    observation_time: str = Field("", description="观测时间")
    observation_weather: str = Field("", description="观测天气")
    people_flow: str = Field("", description="人流量")
    space_type: str = Field("", description="空间类型，如社区、蓝绿、商办")
    sound_type: str = Field("", description="声音类型")
    maintenance_status: str = Field("", description="管理维护状态")
    traffic_flow: str = Field("", description="交通流量")
    description: str = Field("", description="补充描述")

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        aliases = {
            "time_of_day": "observation_time",
            "weather": "observation_weather",
            "crowd_level": "people_flow",
            "location_type": "space_type",
        }
        for old_key, new_key in aliases.items():
            if new_key not in normalized and old_key in normalized:
                normalized[new_key] = normalized[old_key]
        return normalized

    def as_text(self) -> str:
        parts = []
        fields = [
            ("观测时间", self.observation_time),
            ("观测天气", self.observation_weather),
            ("人流量", self.people_flow),
            ("空间类型", self.space_type),
            ("声音类型", self.sound_type),
            ("管理维护状态", self.maintenance_status),
            ("交通流量", self.traffic_flow),
        ]
        parts.extend(f"{label}: {value}" for label, value in fields if value)
        if self.description:
            parts.append(f"描述: {self.description}")
        return "；".join(parts) if parts else ""


class MorphMetrics(BaseModel):
    """七项形态要素指标（比例类多为 0-1，色彩数量为有效色数）。"""

    green_view: float = 0.0
    blue_view: float = 0.0
    sky_view: float = 0.0
    built_ratio: float = 0.0
    color_richness: float = 1.0
    edge_density: float = 0.0
    skyline_variance: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return self.model_dump()

    def as_percent_display(self) -> dict[str, str]:
        pct_keys = {
            "green_view",
            "blue_view",
            "sky_view",
            "built_ratio",
            "edge_density",
            "skyline_variance",
        }
        out = {}
        for k, v in self.as_dict().items():
            if k in pct_keys:
                out[k] = f"{v * 100:.2f}%"
            else:
                out[k] = f"{v:.2f}"
        return out


class TranslationEvidence(BaseModel):
    """Task 2 转换依据的结构化记录。"""

    method: Literal["rule", "rag", "learning", "llm", "expert"]
    summary: str
    reference_id: str = ""
    score: Optional[float] = None


class MorphTranslationResult(BaseModel):
    """翻译官输出：体验旋钮变化 → 形态要素目标"""

    baseline_metrics: MorphMetrics
    target_metrics: MorphMetrics
    delta_from_baseline: dict[str, float] = Field(default_factory=dict)
    experience_baseline: dict[str, float] = Field(default_factory=dict)
    experience_records: list[dict[str, Any]] = Field(default_factory=list)
    experience_targets: dict[str, float] = Field(default_factory=dict)
    experience_delta: dict[str, float] = Field(default_factory=dict)
    rationale: str = ""
    conversion_basis: list[TranslationEvidence] = Field(default_factory=list)
    references_used: list[str] = Field(default_factory=list)
    learning_applied: bool = False


class SpatialObjectAction(BaseModel):
    """空间对象级修改动作。"""

    action: Literal["add", "remove", "adjust"]
    object_type: str
    position: str
    quantity: str = "按目标增量适量配置"
    attributes: list[str] = Field(default_factory=list)
    rationale: str = ""


class ModificationPlan(BaseModel):
    """制图员输出：结构化空间布局方案 + 可执行修改文本。"""

    draft_text: str
    language: str = "en"
    plan_summary: str = ""
    rationale: str = ""
    layout_hints: list[str] = Field(default_factory=list)
    object_actions: list[SpatialObjectAction] = Field(default_factory=list)
    spatial_relations: list[str] = Field(default_factory=list)
    unchanged_regions: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    expert_advice: str = ""
    original_image_path: str = ""
    rag_references: list[str] = Field(default_factory=list)

    @property
    def worldlabs_prompt(self) -> str:
        """World Labs Pano Edit 当前消费的最终修改文本。"""
        return self.draft_text


class GenerationResult(BaseModel):
    """World Labs 文生图结果"""

    output_image_path: str
    prompt_used: str
    mock: bool = False
    raw: dict[str, Any] = Field(default_factory=dict)


class QualityReport(BaseModel):
    """质检员输出"""

    measured_metrics: MorphMetrics
    target_metrics: MorphMetrics
    deviations: dict[str, float]
    passed: bool
    details: str = ""


class LearningFeedback(BaseModel):
    """学习 Agent 反馈记录（翻译准确度）"""

    session_id: str = ""
    experience_baseline: dict[str, float] = Field(default_factory=dict)
    experience_targets: dict[str, float] = Field(default_factory=dict)
    predicted_target_metrics: dict[str, float] = Field(default_factory=dict)
    human_corrected_metrics: dict[str, float] = Field(default_factory=dict)
    accurate: bool = True
    notes: str = ""


class MemoryRecord(BaseModel):
    """知识库记忆条目"""

    id: str
    knobs: dict[str, float] = Field(default_factory=dict)
    experience_baseline: dict[str, float] = Field(default_factory=dict)
    baseline_metrics: dict[str, float] = Field(default_factory=dict)
    target_metrics: dict[str, float] = Field(default_factory=dict)
    scene_context: str = ""
    modification_plan: str = ""
    final_prompt: str = ""
    measured_after: dict[str, float] = Field(default_factory=dict)
    human_corrected_metrics: dict[str, float] = Field(default_factory=dict)
    pre_edit_experience: list[dict[str, Any]] = Field(default_factory=list)
    post_edit_experience: list[dict[str, Any]] = Field(default_factory=list)
    diff_summary: str = ""
    score: Optional[float] = None
    notes: str = ""


# ---------- 兼容旧版（deprecated）----------
class DesignIntent(BaseModel):
    """@deprecated 旧版翻译官输出，请使用 MorphTranslationResult"""

    text: str
    experience_targets: dict[str, float]
    references_used: list[str] = Field(default_factory=list)


class MorphAdjustment(BaseModel):
    """@deprecated 旧版制图员输出，请使用 MorphTranslationResult + ModificationPlan"""

    target_metrics: MorphMetrics
    delta_from_baseline: dict[str, float] = Field(default_factory=dict)
    layout_actions: list[dict[str, Any]] = Field(default_factory=list)
    rationale: str = ""


class PromptDraft(BaseModel):
    """@deprecated 旧版提示词专家输出，请使用 ModificationPlan"""

    draft_prompt: str
    language: str = "en"
    layout_actions: list[dict[str, Any]] = Field(default_factory=list)
