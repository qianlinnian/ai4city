"""
数据模型（全流程共用）
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ExperienceTargets(BaseModel):
    """单人体验感受指标（1-5）"""

    comfort: float = Field(3, ge=1, le=5, description="舒适度")
    restoration: float = Field(3, ge=1, le=5, description="恢复感")
    safety: float = Field(3, ge=1, le=5, description="安全感")
    pleasure: float = Field(3, ge=1, le=5, description="愉悦感")
    stay: float = Field(3, ge=1, le=5, description="可停留意愿")

    def as_dict(self) -> dict[str, float]:
        return self.model_dump()


class PersonExperience(BaseModel):
    """多人体验：单个人的体验指标"""

    person_id: str
    person_name: str = ""
    experience: ExperienceTargets = Field(default_factory=ExperienceTargets)

    def experience_dict(self) -> dict[str, float]:
        return self.experience.as_dict()


class MultiPersonExperience(BaseModel):
    """多人体验指标集合"""

    persons: list[PersonExperience] = Field(default_factory=list)

    def average_experience(self) -> dict[str, float]:
        """计算多人体验均值，作为体验基线参考。"""
        if not self.persons:
            return ExperienceTargets().as_dict()
        keys = ExperienceTargets().as_dict().keys()
        out: dict[str, float] = {}
        for k in keys:
            vals = [p.experience_dict()[k] for p in self.persons]
            out[k] = round(sum(vals) / len(vals), 2)
        return out


class SceneContext(BaseModel):
    """情景要素（与全景图配套）"""

    location_type: str = Field("", description="空间类型，如街巷、广场、口袋公园")
    time_of_day: str = Field("", description="时段，如清晨、午后、傍晚")
    weather: str = Field("", description="天气")
    crowd_level: str = Field("", description="人流密度")
    description: str = Field("", description="补充描述")

    def as_text(self) -> str:
        parts = []
        if self.location_type:
            parts.append(f"空间类型: {self.location_type}")
        if self.time_of_day:
            parts.append(f"时段: {self.time_of_day}")
        if self.weather:
            parts.append(f"天气: {self.weather}")
        if self.crowd_level:
            parts.append(f"人流: {self.crowd_level}")
        if self.description:
            parts.append(f"描述: {self.description}")
        return "；".join(parts) if parts else ""


class MorphMetrics(BaseModel):
    """形态要素指标（比例类多为 0-1，色彩数量为有效色数）"""

    green_view: float = 0.0
    blue_view: float = 0.0
    sky_view: float = 0.0
    built_ratio: float = 0.0
    edge_density: float = 0.0
    color_richness: float = 1.0
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


class MorphTranslationResult(BaseModel):
    """翻译官输出：体验旋钮变化 → 形态要素目标"""

    baseline_metrics: MorphMetrics
    target_metrics: MorphMetrics
    delta_from_baseline: dict[str, float] = Field(default_factory=dict)
    experience_baseline: dict[str, float] = Field(default_factory=dict)
    experience_targets: dict[str, float] = Field(default_factory=dict)
    experience_delta: dict[str, float] = Field(default_factory=dict)
    rationale: str = ""
    references_used: list[str] = Field(default_factory=list)
    learning_applied: bool = False


class ModificationPlan(BaseModel):
    """制图员输出：形态要素目标 → 文生图自然语言修改方案"""

    draft_text: str
    language: str = "en"
    rationale: str = ""
    layout_hints: list[str] = Field(default_factory=list)


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
