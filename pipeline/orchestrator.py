"""
================================================================================
全流程编排器（Orchestrator）v2
文件: pipeline/orchestrator.py
--------------------------------------------------------------------------------
【新流程】
  全景图 + 情景要素
       → 形态解析（SegFormer / fallback）
       → 体验滑块确认 → 翻译官（体验原值→目标值 → 形态目标）  【人工干预①】
       → 制图员（形态目标 → 自然语言方案）                  【人工干预②】
       → World Labs Pano Edit 文生图
       → 质检员（重算指标）
       → 填写修改后多人体验值 → 记忆 Agent + 学习 Agent 入库

【专用工具】
  - morph_metrics_extractor.py   图像解析形态要素
  - worldlabs_agent.py           自然语言文生图

【怎么调用】
  from pipeline.orchestrator import PipelineOrchestrator
  pipe = PipelineOrchestrator()

  state = pipe.start_session(image_path, scene_context={...})
  state = pipe.run_translator(state, experience_targets={...})
  state = pipe.confirm_morph(state, human_metrics={...})
  state = pipe.confirm_plan(state, human_plan="...")
  state = pipe.generate_and_check(state)
  state = pipe.record_post_experience(state, post_experience=[...])
  state = pipe.save_memory(state, score=4)
================================================================================
"""

from __future__ import annotations

import json
import sys
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.cartographer_agent import CartographerAgent
from agents.learning_agent import LearningAgent
from agents.memory_agent import MemoryAgent
from agents.quality_checker_agent import QualityCheckerAgent
from agents.translator_agent import TranslatorAgent
from agents.worldlabs_agent import WorldLabsAgent
from config import SESSION_DIR
from morph_metrics_extractor import MorphMetricsExtractor
from schemas.models import (
    ExperienceTargets,
    ModificationPlan,
    MorphTranslationResult,
    MultiPersonExperience,
    SceneContext,
)


class PipelineOrchestrator:
    def __init__(self, force_metrics_fallback: bool = False):
        self.extractor = MorphMetricsExtractor(force_fallback=force_metrics_fallback)
        self.learning = LearningAgent()
        self.translator = TranslatorAgent()
        self.cartographer = CartographerAgent()
        self.worldlabs = WorldLabsAgent()
        self.quality = QualityCheckerAgent(extractor=self.extractor)
        self.memory = MemoryAgent()

    def start_session(
        self,
        image_path: str | Path,
        scene_context: dict | SceneContext | str | None = None,
        pre_edit_experience: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Step0: 上传全景 + 情景要素，解析形态基线（不调用 Agent）。"""
        image_path = Path(image_path)
        if isinstance(scene_context, SceneContext):
            scene = scene_context
        elif isinstance(scene_context, dict):
            scene = SceneContext(**scene_context)
        else:
            scene = SceneContext(description=str(scene_context or ""))

        baseline = self.extractor.calculate(image_path).as_dict()
        state = {
            "session_id": uuid.uuid4().hex,
            "created_at": datetime.now().isoformat(),
            "image_path": str(image_path),
            "scene_context": scene.model_dump(),
            "scene_context_text": scene.as_text(),
            "pre_edit_experience": pre_edit_experience or [],
            "experience_baseline": None,
            "experience_targets": None,
            "baseline_metrics": baseline,
            "morph_translation": None,
            "confirmed_target_metrics": None,
            "expert_morph_note": "",
            "modification_plan": None,
            "final_prompt": None,
            "expert_plan_edited": False,
            "generation": None,
            "quality_report": None,
            "post_edit_experience": [],
            "memory_id": None,
            "stage": "await_experience_confirm",
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
        """Step1: 体验滑块确认 → 翻译官 → 形态目标（待人工确认）。"""
        state = deepcopy(state)
        targets = ExperienceTargets(**experience_targets).as_dict()
        records = (
            experience_records
            if experience_records is not None
            else state.get("pre_edit_experience") or []
        )
        exp_base = experience_baseline or state.get("experience_baseline")

        result = self.translator.run(
            experience_targets=targets,
            baseline_metrics=state["baseline_metrics"],
            experience_records=records,
            experience_baseline=exp_base,
            scene_context=state.get("scene_context_text", ""),
            original_image_path=state.get("image_path", ""),
        )

        state["pre_edit_experience"] = result.experience_records
        state["experience_baseline"] = result.experience_baseline
        state["experience_targets"] = targets
        state["morph_translation"] = result.model_dump()
        state["confirmed_target_metrics"] = result.target_metrics.model_dump()
        state["stage"] = "await_morph_confirm"
        self._persist(state)
        return state

    def confirm_morph(
        self,
        state: dict[str, Any],
        human_metrics: dict | None = None,
        note: str = "",
        language: str = "en",
    ) -> dict[str, Any]:
        """人工干预①：固化形态目标 → 制图员生成自然语言方案。"""
        state = deepcopy(state)
        if not state.get("morph_translation"):
            raise ValueError("请先 run_translator")

        translation = MorphTranslationResult(**state["morph_translation"])
        predicted = translation.target_metrics.model_dump()
        translation = self.translator.apply_human_override(
            translation, human_metrics=human_metrics, note=note
        )
        state["morph_translation"] = translation.model_dump()
        state["confirmed_target_metrics"] = translation.target_metrics.model_dump()
        state["expert_morph_note"] = note

        # 学习 Agent：记录翻译准确度
        self.learning.record_translation_feedback(
            experience_baseline=state["experience_baseline"],
            experience_targets=state["experience_targets"],
            predicted_target_metrics=predicted,
            human_corrected_metrics=state["confirmed_target_metrics"],
            session_id=state["session_id"],
            notes=note,
        )

        plan = self.cartographer.run(
            baseline_metrics=state["baseline_metrics"],
            target_metrics=state["confirmed_target_metrics"],
            experience_targets=state["experience_targets"],
            experience_baseline=state["experience_baseline"],
            scene_context=state.get("scene_context_text", ""),
            language=language,
            original_image_path=state.get("image_path", ""),
            expert_advice=note,
        )
        state["modification_plan"] = plan.model_dump()
        state["final_prompt"] = plan.draft_text
        state["stage"] = "await_plan_confirm"
        self._persist(state)
        return state

    def confirm_plan(self, state: dict[str, Any], human_plan: str) -> dict[str, Any]:
        """人工干预②：润色自然语言修改方案。"""
        state = deepcopy(state)
        if state.get("modification_plan"):
            original = ModificationPlan(**state["modification_plan"])
            edited = self.cartographer.apply_human_edit(original, human_plan)
            state["modification_plan"] = edited.model_dump()
            state["expert_plan_edited"] = edited.draft_text != original.draft_text
            state["final_prompt"] = edited.worldlabs_prompt
        else:
            state["final_prompt"] = human_plan.strip()
            state["expert_plan_edited"] = True
        state["stage"] = "ready_to_generate"
        self._persist(state)
        return state

    def generate_and_check(self, state: dict[str, Any]) -> dict[str, Any]:
        """Step3-4: World Labs 文生图 + 质检。"""
        state = deepcopy(state)
        prompt = state.get("final_prompt")
        if not prompt:
            raise ValueError("缺少修改方案，请先 confirm_morph / confirm_plan")

        gen = self.worldlabs.run(state["image_path"], prompt)
        state["generation"] = gen.model_dump()

        report = self.quality.run(
            gen.output_image_path,
            state["confirmed_target_metrics"],
        )
        state["quality_report"] = report.model_dump()
        state["stage"] = "await_post_experience"
        self._persist(state)
        return state

    def record_post_experience(
        self,
        state: dict[str, Any],
        post_experience: list[dict] | MultiPersonExperience,
    ) -> dict[str, Any]:
        """修改后全景：记录多人体验指标。"""
        state = deepcopy(state)
        if isinstance(post_experience, MultiPersonExperience):
            persons = [p.model_dump() for p in post_experience.persons]
        else:
            persons = post_experience
        state["post_edit_experience"] = persons
        self._persist(state)
        return state

    def save_memory(
        self,
        state: dict[str, Any],
        human_corrected_metrics: dict | None = None,
        score: float | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        """写入知识库。"""
        state = deepcopy(state)
        qr = state.get("quality_report") or {}
        measured = (qr.get("measured_metrics") or {}) if qr else {}

        plan_text = ""
        if state.get("modification_plan"):
            plan_text = state["modification_plan"].get("draft_text", "")

        mem = self.memory.store_feedback(
            knobs=state.get("experience_targets") or {},
            experience_baseline=state.get("experience_baseline") or {},
            baseline_metrics=state.get("baseline_metrics"),
            target_metrics=state.get("confirmed_target_metrics"),
            scene_context=state.get("scene_context_text", ""),
            modification_plan=plan_text,
            final_prompt=state.get("final_prompt") or "",
            measured_after=measured,
            human_corrected_metrics=human_corrected_metrics or {},
            pre_edit_experience=state.get("pre_edit_experience") or [],
            post_edit_experience=state.get("post_edit_experience") or [],
            score=score,
            notes=notes,
        )
        state["memory_id"] = mem.id
        state["stage"] = "done"
        self._persist(state)
        return state

    def _persist(self, state: dict[str, Any]) -> None:
        path = SESSION_DIR / f"{state['session_id']}.json"
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def run_full_demo(
    image_path: str,
    experience_targets: dict | None = None,
    scene_context: dict | None = None,
) -> dict[str, Any]:
    """无人工干预的一键演示。"""
    experience_targets = experience_targets or {
        "comfort": 4,
        "naturalness": 4,
        "safety": 4,
        "relaxation": 4,
        "environmental_disturbance": 2,
        "stay_intention": 4,
        "overall_impression": 4,
    }
    scene_context = scene_context or {
        "space_type": "社区",
        "observation_time": "午后",
        "observation_weather": "晴",
    }
    pipe = PipelineOrchestrator(force_metrics_fallback=True)
    state = pipe.start_session(
        image_path,
        scene_context=scene_context,
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
    state = pipe.run_translator(state, experience_targets)
    state = pipe.confirm_morph(state)
    state = pipe.confirm_plan(state, state["final_prompt"])
    state = pipe.generate_and_check(state)
    state = pipe.save_memory(state, score=4, notes="auto demo")
    return state


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="全流程一键演示")
    parser.add_argument("image", help="全景图路径")
    parser.add_argument("--fallback", action="store_true")
    args = parser.parse_args()
    result = run_full_demo(args.image)
    print(
        json.dumps(
            {
                "session_id": result["session_id"],
                "stage": result["stage"],
                "memory_id": result["memory_id"],
                "output": (result.get("generation") or {}).get("output_image_path"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
