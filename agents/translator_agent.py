"""
================================================================================
翻译官 Agent（Translator Agent）
文件: agents/translator_agent.py
--------------------------------------------------------------------------------
【角色】
  接收前端体验感受滑块的「原值 → 目标值」变化，结合映射规则、情景要素、
  本地知识库与（可选）学习 Agent 的多轮学习结果，将体验调节翻译为
  「形态要素原值 + 形态要素目标值」。

【输入】
  - experience_baseline: dict   体验原值（默认 3；可由多人体验均值覆盖）
  - experience_targets: dict    体验目标值（前端五个滑块确认后的值）
  - baseline_metrics: dict      图像解析得到的形态要素基线（7 维）
  - scene_context: str          情景要素文本（可选）
  - learning_agent: LearningAgent | None  可选学习 Agent

【输出】
  - MorphTranslationResult
      .baseline_metrics        原先形态要素（来自图像解析）
      .target_metrics          计算后的形态要素目标
      .delta_from_baseline     形态增量
      .experience_baseline     体验原值
      .experience_targets      体验目标
      .experience_delta         体验变化量
      .rationale                翻译理由
      .references_used          知识库检索 id
      .learning_applied         是否应用了学习 Agent 修正

【输出到哪里】
  → 前端展示「原形态 / 目标形态」，供人工干预修改目标值
  → 确认后传给「制图员 Agent」(cartographer_agent.run)

【怎么调用】
  from agents.translator_agent import TranslatorAgent
  agent = TranslatorAgent()
  result = agent.run(
      experience_baseline={"comfort": 3, ...},
      experience_targets={"comfort": 4, ...},
      baseline_metrics={...},
      scene_context="街巷；午后",
  )
================================================================================
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import TYPE_CHECKING

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import EXPERIENCE_KEYS, EXPERIENCE_LABELS_ZH, MORPH_BOUNDS, MORPH_KEYS, MORPH_LABELS_ZH
from knowledge_base.kb_store import KnowledgeBase, kb as default_kb
from schemas.models import ExperienceTargets, MorphMetrics, MorphTranslationResult
from utils import llm_client

if TYPE_CHECKING:
    from agents.learning_agent import LearningAgent


SYSTEM_PROMPT = (
    "你是城市微空间体验-形态翻译官。根据体验感受从原值到目标值的变化，"
    "在已有形态基线上给出合理的形态要素目标数值。"
    "只输出 JSON："
    '{"target_metrics":{7个形态键},"rationale":"简要理由"}'
)


class TranslatorAgent:
    def __init__(
        self,
        knowledge_base: KnowledgeBase | None = None,
        learning_agent: LearningAgent | None = None,
    ):
        self.kb = knowledge_base or default_kb
        self.learning = learning_agent

    def run(
        self,
        experience_targets: dict | ExperienceTargets,
        baseline_metrics: dict,
        experience_baseline: dict | None = None,
        scene_context: str = "",
        learning_agent: LearningAgent | None = None,
    ) -> MorphTranslationResult:
        if isinstance(experience_targets, ExperienceTargets):
            targets = experience_targets.as_dict()
        else:
            targets = ExperienceTargets(**experience_targets).as_dict()

        exp_base = experience_baseline or {k: 3.0 for k in EXPERIENCE_KEYS}
        exp_base = {k: float(exp_base.get(k, 3.0)) for k in EXPERIENCE_KEYS}
        exp_delta = {k: round(targets[k] - exp_base[k], 2) for k in EXPERIENCE_KEYS}

        morph_base = self._normalize_baseline(baseline_metrics)
        refs = self.kb.retrieve_similar(targets, top_k=2)
        ref_ids = [r.get("id", "") for r in refs if r.get("id")]

        # 1) 规则映射
        target = self._rule_map_targets(targets, exp_base, morph_base)

        # 2) 知识库相似案例融合
        target = self._blend_with_memory(target, refs)

        # 3) 可选学习 Agent 修正
        learning = learning_agent or self.learning
        learning_applied = False
        if learning is not None:
            learned = learning.get_morph_correction(
                experience_baseline=exp_base,
                experience_targets=targets,
                baseline_metrics=morph_base,
                predicted_target=target,
            )
            if learned:
                for k, v in learned.items():
                    if k in target:
                        target[k] = self._clamp(k, float(v))
                learning_applied = True

        # 4) LLM 细化（失败则保留规则结果）
        rationale = self._refine_via_llm(
            exp_base, targets, exp_delta, morph_base, target, scene_context, refs
        )

        delta = {k: round(target[k] - morph_base[k], 4) for k in MORPH_KEYS}

        return MorphTranslationResult(
            baseline_metrics=MorphMetrics(**{k: morph_base[k] for k in MORPH_KEYS}),
            target_metrics=MorphMetrics(**{k: target[k] for k in MORPH_KEYS}),
            delta_from_baseline=delta,
            experience_baseline=exp_base,
            experience_targets=targets,
            experience_delta=exp_delta,
            rationale=rationale,
            references_used=ref_ids,
            learning_applied=learning_applied,
        )

    def apply_human_override(
        self,
        result: MorphTranslationResult,
        human_metrics: dict | None = None,
        note: str = "",
    ) -> MorphTranslationResult:
        """前端人工干预：覆盖形态要素目标值。"""
        data = result.model_dump()
        base = data["baseline_metrics"]
        if human_metrics:
            merged = {**data["target_metrics"], **human_metrics}
            data["target_metrics"] = MorphMetrics(
                **{k: merged[k] for k in MORPH_KEYS}
            ).model_dump()
            data["delta_from_baseline"] = {
                k: round(data["target_metrics"][k] - base[k], 4) for k in MORPH_KEYS
            }
        if note:
            data["rationale"] = (data.get("rationale") or "") + f" | 人工备注: {note}"
        return MorphTranslationResult(**data)

    # ------------------------------------------------------------------ #
    def _normalize_baseline(self, baseline: dict) -> dict[str, float]:
        defaults = {
            "green_view": 0.15,
            "blue_view": 0.02,
            "sky_view": 0.25,
            "built_ratio": 0.45,
            "edge_density": 0.08,
            "color_richness": 3.5,
            "skyline_variance": 0.03,
        }
        out = dict(defaults)
        for k in MORPH_KEYS:
            if k in baseline and baseline[k] is not None:
                out[k] = float(baseline[k])
        return out

    def _clamp(self, key: str, value: float) -> float:
        lo, hi = MORPH_BOUNDS[key]
        return float(min(hi, max(lo, value)))

    def _rule_map_targets(
        self,
        targets: dict,
        exp_base: dict,
        morph_base: dict,
    ) -> dict[str, float]:
        """依据 mapping_rules.json 将体验变化映射到形态增量。"""
        rules = self.kb.get_mapping_rules().get("rules", [])
        result = copy.deepcopy(morph_base)

        for rule in rules:
            high_key = rule.get("when_high")
            low_key = rule.get("when_low")
            fire = False
            strength = 0.0

            if high_key and float(targets.get(high_key, 3)) >= 4:
                delta_e = float(targets[high_key]) - float(exp_base.get(high_key, 3))
                if delta_e > 0:
                    fire = True
                    strength = max(strength, delta_e / 2.0)
            if low_key and float(targets.get(low_key, 3)) <= 2:
                delta_e = float(exp_base.get(low_key, 3)) - float(targets[low_key])
                if delta_e > 0:
                    fire = True
                    strength = max(strength, delta_e / 2.0)

            if not fire:
                continue

            strength = max(strength, 0.3)
            for mk, delta in (rule.get("adjust") or {}).items():
                if mk in result:
                    result[mk] = result[mk] + float(delta) * strength

        return {k: self._clamp(k, result[k]) for k in MORPH_KEYS}

    def _blend_with_memory(self, target: dict, refs: list[dict]) -> dict[str, float]:
        mem_targets = [r.get("target_metrics") for r in refs if r.get("target_metrics")]
        if not mem_targets:
            return target
        blended = dict(target)
        for k in MORPH_KEYS:
            vals = [float(t[k]) for t in mem_targets if k in t]
            if vals:
                mem_avg = sum(vals) / len(vals)
                blended[k] = self._clamp(k, 0.7 * target[k] + 0.3 * mem_avg)
        return blended

    def _refine_via_llm(
        self,
        exp_base: dict,
        targets: dict,
        exp_delta: dict,
        morph_base: dict,
        target: dict,
        scene_context: str,
        refs: list[dict],
    ) -> str:
        exp_lines = [
            f"{EXPERIENCE_LABELS_ZH.get(k, k)}: {exp_base[k]}→{targets[k]} (Δ{exp_delta[k]:+.1f})"
            for k in EXPERIENCE_KEYS
        ]
        morph_lines = [
            f"{MORPH_LABELS_ZH.get(k, k)}: {morph_base[k]:.3f}→{target[k]:.3f}"
            for k in MORPH_KEYS
        ]
        fewshot = ""
        if refs:
            fewshot = "\n".join(
                f"案例: knobs={r.get('knobs')}; target={r.get('target_metrics')}"
                for r in refs[:2]
            )

        user = (
            f"【体验变化】\n" + "\n".join(exp_lines) + "\n"
            f"【形态基线→规则目标】\n" + "\n".join(morph_lines) + "\n"
        )
        if scene_context:
            user += f"【情景要素】{scene_context}\n"
        if fewshot:
            user += f"【知识库】\n{fewshot}\n"

        raw = llm_client.chat(SYSTEM_PROMPT, user)
        if raw:
            try:
                import json

                text = raw.strip()
                if "```" in text:
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                data = json.loads(text)
                if isinstance(data.get("target_metrics"), dict):
                    for k, v in data["target_metrics"].items():
                        if k in target:
                            target[k] = self._clamp(k, float(v))
                if data.get("rationale"):
                    return str(data["rationale"])
            except Exception as e:
                print(f"[Translator] LLM JSON 解析失败: {e}")

        return self._rule_rationale(exp_base, targets, exp_delta, target)

    def _rule_rationale(
        self,
        exp_base: dict,
        targets: dict,
        exp_delta: dict,
        target: dict,
    ) -> str:
        changes = [
            f"{EXPERIENCE_LABELS_ZH.get(k, k)}{exp_base[k]}→{targets[k]}"
            for k in EXPERIENCE_KEYS
            if abs(exp_delta[k]) >= 0.5
        ]
        morph_focus = [
            f"{MORPH_LABELS_ZH[k]}目标{target[k]:.2f}"
            for k in ("green_view", "sky_view", "built_ratio")
        ]
        focus = "、".join(changes) or "均衡微调"
        return (
            f"依据体验变化（{focus}）与环境心理学映射规则，"
            f"建议形态目标：{', '.join(morph_focus)}。"
        )


def run_translator(
    experience_targets: dict,
    baseline_metrics: dict,
    experience_baseline: dict | None = None,
    scene_context: str = "",
) -> MorphTranslationResult:
    return TranslatorAgent().run(
        experience_targets=experience_targets,
        baseline_metrics=baseline_metrics,
        experience_baseline=experience_baseline,
        scene_context=scene_context,
    )


if __name__ == "__main__":
    demo = TranslatorAgent().run(
        experience_baseline={"comfort": 3, "restoration": 3, "safety": 3, "pleasure": 3, "stay": 3},
        experience_targets={"comfort": 4, "restoration": 5, "safety": 3, "pleasure": 4, "stay": 4},
        baseline_metrics={"green_view": 0.12, "sky_view": 0.28, "built_ratio": 0.5},
        scene_context="高密度街巷；午后晴天",
    )
    print(demo.model_dump_json(indent=2, ensure_ascii=False))
