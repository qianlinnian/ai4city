"""Orchestration for the Excel-driven panorama optimisation workflow.

Task 1 runs offline and writes metrics into the project data table.  The online
pipeline starts with an image plus the seven baseline values already selected
from that table.  Image generation is injected so this module does not depend
on a particular UI or paid service.
"""

from __future__ import annotations

import json
import sys
import uuid
from collections.abc import Callable, Mapping
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.cartographer_agent import CartographerAgent
from agents.learning_agent import LearningAgent
from agents.memory_agent import MemoryAgent
from agents.quality_checker_agent import QualityCheckerAgent, validate_morph_metrics
from agents.scene_understanding_agent import SceneUnderstandingAgent
from agents.translator_agent import TranslatorAgent
from config import SESSION_DIR
from schemas.models import (
    ExperienceTargets,
    GenerationResult,
    ModificationPlan,
    MorphTranslationResult,
    MultiPersonExperience,
    SceneContext,
)


INPUT_PENDING = "input_pending"
MORPH_REVIEW = "morph_review"
PLAN_REVIEW = "plan_review"
PLAN_CONFIRMED = "plan_confirmed"
GENERATED = "generated"
VALIDATION_PENDING = "validation_pending"
COMPLETED = "completed"


class GeneratorProtocol(Protocol):
    """Minimal object interface accepted by :class:`PipelineOrchestrator`."""

    def run(self, *, image_path: str | Path, prompt: str) -> GenerationResult: ...


GeneratorCallable = Callable[[str | Path, str], GenerationResult]
PostEditMetricsExtractor = Callable[[str | Path], Mapping[str, float]]


class PipelineOrchestrator:
    def __init__(
        self,
        generator: GeneratorProtocol | GeneratorCallable | None = None,
        *,
        force_metrics_fallback: bool | None = None,
        translator: TranslatorAgent | None = None,
        cartographer: CartographerAgent | None = None,
        learning: LearningAgent | None = None,
        quality: QualityCheckerAgent | None = None,
        memory: MemoryAgent | None = None,
        scene_understander: SceneUnderstandingAgent | None = None,
        post_edit_metrics_extractor: PostEditMetricsExtractor | None = None,
    ) -> None:
        # 保留历史关键字以兼容既有 UI；生成后指标复算由显式注入的
        # ``post_edit_metrics_extractor`` 控制，而不是由此参数隐式决定。
        _ = force_metrics_fallback
        if generator is None and force_metrics_fallback is not None:
            from app.generation_backend import generate

            generator = generate
        self.generator = generator
        self.learning = learning or LearningAgent()
        self.translator = translator or TranslatorAgent()
        self.cartographer = cartographer or CartographerAgent()
        self.quality = quality or QualityCheckerAgent()
        self.memory = memory or MemoryAgent()
        self.scene_understander = scene_understander or SceneUnderstandingAgent()
        self.post_edit_metrics_extractor = post_edit_metrics_extractor

    def start_session(
        self,
        image_path: str | Path,
        baseline_metrics: Mapping[str, float] | None = None,
        scene_context: dict | SceneContext | str | None = None,
        pre_edit_experience: list[dict] | None = None,
        *,
        image_name: str | None = None,
        skip_extract: bool | None = None,
    ) -> dict[str, Any]:
        """Create a session from an image and seven values read from Excel."""

        # Compatibility-only arguments used by the existing ``main`` UI.
        # No image metric extraction is performed regardless of their value.
        _ = skip_extract
        source = Path(image_path)
        if not source.is_file():
            raise FileNotFoundError(f"找不到原始全景图：{source}")
        if baseline_metrics is None:
            raise ValueError("缺少从项目大表读取的七项形态基线指标")
        baseline = validate_morph_metrics(baseline_metrics, field_name="基线指标")

        if isinstance(scene_context, SceneContext):
            scene = scene_context
        elif isinstance(scene_context, dict):
            scene = SceneContext(**scene_context)
        else:
            scene = SceneContext(description=str(scene_context or ""))

        state = {
            "session_id": uuid.uuid4().hex,
            "created_at": datetime.now().isoformat(),
            "image_path": str(source.resolve()),
            "image_name": image_name or source.name,
            "scene_context": scene.model_dump(),
            "scene_context_text": scene.as_text(),
            "pre_edit_experience": pre_edit_experience or [],
            "experience_baseline": None,
            "experience_targets": None,
            "baseline_metrics": baseline,
            "morph_translation": None,
            "translator_prompt_variant": None,
            "translator_round": 0,
            "panorama_views": [],
            "scene_understanding": None,
            "task2_reasonableness": None,
            "confirmed_target_metrics": None,
            "expert_morph_note": "",
            "modification_plan": None,
            "cartographer_scene_profile": None,
            "final_prompt": None,
            "expert_plan_edited": False,
            "generation": None,
            "quality_report": None,
            "post_edit_metrics": None,
            "post_edit_metrics_error": None,
            "post_edit_experience": [],
            "memory_id": None,
            "stage": INPUT_PENDING,
        }
        self._persist(state)
        return state

    def run_translator(
        self,
        state: dict[str, Any],
        experience_targets: dict,
        experience_records: list[dict] | None = None,
        experience_baseline: dict | None = None,
    ) -> dict[str, Any]:
        """Confirm experience targets and produce morphology targets for review."""

        self._require_stage(state, INPUT_PENDING, MORPH_REVIEW)
        updated = deepcopy(state)
        targets = ExperienceTargets(**experience_targets).as_dict()
        records = (
            experience_records
            if experience_records is not None
            else updated.get("pre_edit_experience") or []
        )
        exp_base = experience_baseline or updated.get("experience_baseline")
        previous_translation = updated.get("morph_translation") or {}
        previous_experience_targets = updated.get("experience_targets")
        is_revision = bool(previous_translation and previous_experience_targets)
        prompt_variant = "revision" if is_revision else "initial"
        if not updated.get("scene_understanding"):
            inventory = self.scene_understander.run(
                updated.get("image_path", ""),
                image_id=Path(updated.get("image_path", "")).stem,
            )
            updated["scene_understanding"] = inventory.model_dump()
            updated["panorama_views"] = [
                view.model_dump() for view in inventory.view_metadata
            ]
        result = self.translator.run(
            experience_targets=targets,
            baseline_metrics=updated["baseline_metrics"],
            experience_records=records,
            experience_baseline=exp_base,
            scene_context=updated.get("scene_context_text", ""),
            original_image_path=updated.get("image_path", ""),
            scene_understanding=updated.get("scene_understanding"),
            prompt_variant=prompt_variant,
            previous_experience_targets=previous_experience_targets,
            previous_target_metrics=previous_translation.get("target_metrics"),
        )

        updated["pre_edit_experience"] = result.experience_records
        updated["experience_baseline"] = result.experience_baseline
        updated["experience_targets"] = targets
        updated["morph_translation"] = result.model_dump()
        updated["translator_prompt_variant"] = prompt_variant
        updated["translator_round"] = int(updated.get("translator_round") or 0) + 1
        updated["confirmed_target_metrics"] = result.target_metrics.model_dump()
        updated["task2_reasonableness"] = deepcopy(
            getattr(self.translator, "last_reasonableness_report", {})
        )
        updated["stage"] = MORPH_REVIEW
        self._persist(updated)
        return updated

    def confirm_morph(
        self,
        state: dict[str, Any],
        human_metrics: dict | None = None,
        note: str = "",
        language: str = "en",
    ) -> dict[str, Any]:
        """Confirm morphology targets and produce a structured layout plan."""

        self._require_stage(state, MORPH_REVIEW)
        updated = deepcopy(state)
        if not updated.get("morph_translation"):
            raise ValueError("缺少翻译官结果，请先运行翻译官")

        translation = MorphTranslationResult(**updated["morph_translation"])
        predicted = translation.target_metrics.model_dump()
        translation = self.translator.apply_human_override(
            translation, human_metrics=human_metrics, note=note
        )
        updated["morph_translation"] = translation.model_dump()
        updated["confirmed_target_metrics"] = translation.target_metrics.model_dump()
        updated["expert_morph_note"] = note

        self.learning.record_translation_feedback(
            experience_baseline=updated["experience_baseline"],
            experience_targets=updated["experience_targets"],
            predicted_target_metrics=predicted,
            human_corrected_metrics=updated["confirmed_target_metrics"],
            session_id=updated["session_id"],
            notes=note,
        )
        plan = self.cartographer.run(
            baseline_metrics=updated["baseline_metrics"],
            target_metrics=updated["confirmed_target_metrics"],
            experience_targets=updated["experience_targets"],
            experience_baseline=updated["experience_baseline"],
            scene_context=updated.get("scene_context_text", ""),
            language=language,
            original_image_path=updated.get("image_path", ""),
            expert_advice=note,
            scene_understanding=updated.get("scene_understanding"),
            scene_type=str((updated.get("scene_context") or {}).get("space_type", "")),
        )
        updated["modification_plan"] = plan.model_dump()
        updated["cartographer_scene_profile"] = plan.scene_prompt_profile
        updated["final_prompt"] = plan.draft_text
        updated["stage"] = PLAN_REVIEW
        self._persist(updated)
        return updated

    def confirm_plan(self, state: dict[str, Any], human_plan: str) -> dict[str, Any]:
        """Confirm the editable natural-language generation prompt."""

        self._require_stage(state, PLAN_REVIEW)
        clean_plan = str(human_plan).strip()
        if not clean_plan:
            raise ValueError("空间布局方案不能为空")
        updated = deepcopy(state)
        if updated.get("modification_plan"):
            original = ModificationPlan(**updated["modification_plan"])
            edited = self.cartographer.apply_human_edit(original, clean_plan)
            updated["modification_plan"] = edited.model_dump()
            updated["expert_plan_edited"] = edited.draft_text != original.draft_text
            updated["final_prompt"] = edited.worldlabs_prompt
        else:  # Defensive compatibility with restored early sessions.
            updated["final_prompt"] = clean_plan
            updated["expert_plan_edited"] = True
        updated["stage"] = PLAN_CONFIRMED
        self._persist(updated)
        return updated

    def generate_and_check(
        self,
        state: dict[str, Any],
        modified_metrics: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        """Run the generator, then calculate/record the seven post-edit metrics.

        A supplied ``modified_metrics`` takes precedence for API or batch callers.
        When a configured extractor fails, the generated image remains recorded and
        the visible error is persisted instead of fabricating quality values.
        """

        self._require_stage(state, PLAN_CONFIRMED)
        if self.generator is None:
            raise RuntimeError(
                "未配置图像生成器；请注入 generator，或由前端生成后调用 record_generation"
            )
        result = self._run_generator(state["image_path"], state.get("final_prompt") or "")
        updated = self.record_generation(state, result, modified_metrics=modified_metrics)
        if modified_metrics is not None or self.post_edit_metrics_extractor is None:
            return updated
        try:
            measured = self.post_edit_metrics_extractor(result.output_image_path)
        except Exception as exc:
            failed = deepcopy(updated)
            failed["post_edit_metrics_error"] = str(exc)
            self._persist(failed)
            return failed
        return self.record_quality_metrics(updated, measured)

    def record_generation(
        self,
        state: dict[str, Any],
        generation: GenerationResult | Mapping[str, Any],
        *,
        modified_metrics: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        """Record a result created by the UI's selected generation backend."""

        self._require_stage(state, PLAN_CONFIRMED)
        updated = deepcopy(state)
        result = (
            generation
            if isinstance(generation, GenerationResult)
            else GenerationResult(**dict(generation))
        )
        updated["generation"] = result.model_dump()
        updated["post_edit_metrics_error"] = None
        updated["stage"] = GENERATED
        if modified_metrics is not None:
            updated = self.record_quality_metrics(updated, modified_metrics)
        else:
            self._persist(updated)
        return updated

    def record_quality_metrics(
        self,
        state: dict[str, Any],
        modified_metrics: Mapping[str, float],
        thresholds: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        """Compare the optional post-edit row read from the metrics table."""

        self._require_stage(state, GENERATED, VALIDATION_PENDING)
        if not state.get("confirmed_target_metrics"):
            raise ValueError("缺少已确认的形态目标")
        updated = deepcopy(state)
        report = self.quality.run(
            modified_metrics,
            updated["confirmed_target_metrics"],
            thresholds,
        )
        updated["post_edit_metrics"] = report.measured_metrics.as_dict()
        updated["quality_report"] = report.model_dump()
        updated["post_edit_metrics_error"] = None
        updated["stage"] = VALIDATION_PENDING
        self._persist(updated)
        return updated

    def record_post_experience(
        self,
        state: dict[str, Any],
        post_experience: list[dict] | MultiPersonExperience,
    ) -> dict[str, Any]:
        """Record post-edit participant ratings without requiring quality metrics."""

        self._require_stage(state, GENERATED, VALIDATION_PENDING)
        updated = deepcopy(state)
        if isinstance(post_experience, MultiPersonExperience):
            persons = [person.model_dump() for person in post_experience.persons]
        else:
            persons = MultiPersonExperience(persons=post_experience).model_dump()["persons"]
        updated["post_edit_experience"] = persons
        updated["stage"] = VALIDATION_PENDING
        self._persist(updated)
        return updated

    def save_memory(
        self,
        state: dict[str, Any],
        human_corrected_metrics: dict | None = None,
        score: float | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        """Persist feedback; post-edit metrics remain optional."""

        self._require_stage(state, GENERATED, VALIDATION_PENDING)
        updated = deepcopy(state)
        quality_report = updated.get("quality_report") or {}
        measured = quality_report.get("measured_metrics") or {}
        plan_text = ""
        if updated.get("modification_plan"):
            plan_text = updated["modification_plan"].get("draft_text", "")

        memory = self.memory.store_feedback(
            knobs=updated.get("experience_targets") or {},
            experience_baseline=updated.get("experience_baseline") or {},
            baseline_metrics=updated.get("baseline_metrics"),
            target_metrics=updated.get("confirmed_target_metrics"),
            scene_context=updated.get("scene_context_text", ""),
            modification_plan=plan_text,
            final_prompt=updated.get("final_prompt") or "",
            measured_after=measured,
            human_corrected_metrics=human_corrected_metrics or {},
            pre_edit_experience=updated.get("pre_edit_experience") or [],
            post_edit_experience=updated.get("post_edit_experience") or [],
            score=score,
            notes=notes,
        )
        updated["memory_id"] = memory.id
        updated["stage"] = COMPLETED
        self._persist(updated)
        return updated

    def _run_generator(self, image_path: str, prompt: str) -> GenerationResult:
        if not prompt.strip():
            raise ValueError("缺少已确认的空间布局方案")
        generator = self.generator
        if generator is None:  # pragma: no cover - guarded by caller
            raise RuntimeError("未配置图像生成器")
        if hasattr(generator, "run"):
            result = generator.run(image_path=image_path, prompt=prompt)  # type: ignore[union-attr]
        else:
            result = generator(image_path, prompt)  # type: ignore[operator]
        if isinstance(result, GenerationResult):
            return result
        if isinstance(result, Mapping):
            return GenerationResult(**dict(result))
        raise TypeError("生成器必须返回 GenerationResult 或兼容字典")

    @staticmethod
    def _require_stage(state: Mapping[str, Any], *allowed: str) -> None:
        current = state.get("stage")
        if current not in allowed:
            expected = "、".join(allowed)
            raise ValueError(f"当前阶段 {current!r} 不允许执行此操作；需要阶段：{expected}")

    @staticmethod
    def _persist(state: Mapping[str, Any]) -> None:
        path = SESSION_DIR / f"{state['session_id']}.json"
        path.write_text(
            json.dumps(dict(state), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def run_full_demo(
    image_path: str,
    baseline_metrics: Mapping[str, float],
    *,
    generator: GeneratorProtocol | GeneratorCallable,
    experience_targets: dict | None = None,
    scene_context: dict | None = None,
) -> dict[str, Any]:
    """Run a no-human-intervention demo with explicit metrics and generator."""

    targets = experience_targets or {
        "comfort": 4,
        "naturalness": 4,
        "safety": 4,
        "relaxation": 4,
        "environmental_disturbance": 2,
        "stay_intention": 4,
        "overall_impression": 4,
    }
    context = scene_context or {
        "space_type": "社区",
        "observation_time": "午后",
        "observation_weather": "晴",
    }
    pipe = PipelineOrchestrator(generator=generator)
    state = pipe.start_session(
        image_path,
        baseline_metrics,
        scene_context=context,
        pre_edit_experience=[
            {
                "person_id": "demo-1",
                "person_name": "演示参与者",
                "experience": {
                    "comfort": 3,
                    "naturalness": 3,
                    "safety": 3,
                    "relaxation": 3,
                    "environmental_disturbance": 3,
                    "stay_intention": 3,
                    "overall_impression": 3,
                },
            }
        ],
    )
    state = pipe.run_translator(state, targets)
    state = pipe.confirm_morph(state)
    state = pipe.confirm_plan(state, state["final_prompt"])
    state = pipe.generate_and_check(state)
    state = pipe.save_memory(state, score=4, notes="auto demo")
    return state


__all__ = [
    "COMPLETED",
    "GENERATED",
    "GeneratorProtocol",
    "INPUT_PENDING",
    "MORPH_REVIEW",
    "PLAN_CONFIRMED",
    "PLAN_REVIEW",
    "PipelineOrchestrator",
    "VALIDATION_PENDING",
    "run_full_demo",
]
