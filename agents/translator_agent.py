"""
================================================================================
翻译官 Agent（Translator Agent）
文件: agents/translator_agent.py
--------------------------------------------------------------------------------
【角色】
  接收前端七项体验感受的「原值 → 目标值」变化，结合原始全景、映射规则、情景要素、
  本地知识库与（可选）学习 Agent 的多轮学习结果，将体验调节翻译为
  「形态要素原值 + 形态要素目标值」。

【输入】
  - experience_baseline: dict   体验原值（默认 3；可由多人体验均值覆盖）
  - experience_targets: dict    七项体验目标值（前端确认后的值）
  - baseline_metrics: dict      图像解析得到的形态要素基线（7 维）
  - original_image_path: str    原始全景路径（可选，用于多模态 LLM）
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
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    EXPERIENCE_DIRECTIONS,
    EXPERIENCE_KEYS,
    EXPERIENCE_LABELS_ZH,
    MORPH_BOUNDS,
    MORPH_KEYS,
    MORPH_LABELS_ZH,
    normalize_experience_values,
)
from knowledge_base.kb_store import KnowledgeBase, kb as default_kb
from schemas.models import (
    ExperienceTargets,
    MorphMetrics,
    MorphTranslationResult,
    TranslationEvidence,
)
from utils import llm_client

if TYPE_CHECKING:
    from agents.learning_agent import LearningAgent


SYSTEM_PROMPT = (
    "你是城市微空间体验-形态翻译官。根据七项VR体感从原值到目标值的变化，"
    "结合原始全景、情景要素、启发式参数与RAG案例，在已有形态基线上给出合理的"
    "七项形态要素目标数值。环境干扰感是反向指标，目标越低表示期望干扰越少。"
    "不得凭空改变建筑与道路等不可逆结构。"
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
        original_image_path: str = "",
        learning_agent: LearningAgent | None = None,
    ) -> MorphTranslationResult:
        if isinstance(experience_targets, ExperienceTargets):
            targets = experience_targets.as_dict()
        else:
            targets = ExperienceTargets(**experience_targets).as_dict()

        exp_base = normalize_experience_values(experience_baseline)
        exp_delta = {k: round(targets[k] - exp_base[k], 2) for k in EXPERIENCE_KEYS}

        morph_base = self._normalize_baseline(baseline_metrics)
        refs = self.kb.retrieve_experience_cases(
            exp_base,
            targets,
            scene_context=scene_context,
            top_k=2,
        )
        ref_ids = [r.get("id", "") for r in refs if r.get("id")]

        # 1) 规则映射
        target, rule_contributions = self._rule_map_targets(
            targets, exp_base, morph_base
        )
        conversion_basis = [
            TranslationEvidence(
                method="rule",
                summary=self._format_rule_contributions(rule_contributions),
            )
        ]

        # 2) 知识库相似案例融合
        target = self._blend_with_memory(target, morph_base, refs)
        conversion_basis.extend(
            TranslationEvidence(
                method="rag",
                reference_id=str(ref.get("id", "")),
                score=float(ref["_rag_score"])
                if ref.get("_rag_score") is not None
                else None,
                summary="按案例形态增量参与 25% 权重融合",
            )
            for ref in refs
            if ref.get("id")
        )

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
                conversion_basis.append(
                    TranslationEvidence(
                        method="learning",
                        summary="应用已启用的历史专家纠偏结果",
                    )
                )

        # 4) LLM 细化（失败则保留规则结果）
        rationale, llm_applied = self._refine_via_llm(
            exp_base,
            targets,
            exp_delta,
            morph_base,
            target,
            scene_context,
            refs,
            original_image_path,
        )
        if llm_applied:
            conversion_basis.append(
                TranslationEvidence(
                    method="llm",
                    summary="多模态模型在规则/RAG结果上细化七项形态目标",
                )
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
            conversion_basis=conversion_basis,
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
            self._validate_morph_values(
                human_metrics,
                source="专家形态目标",
                require_complete=False,
            )
            merged = {**data["target_metrics"], **human_metrics}
            data["target_metrics"] = MorphMetrics(
                **{k: merged[k] for k in MORPH_KEYS}
            ).model_dump()
            data["delta_from_baseline"] = {
                k: round(data["target_metrics"][k] - base[k], 4) for k in MORPH_KEYS
            }
        if note:
            data["rationale"] = (data.get("rationale") or "") + f" | 人工备注: {note}"
        if human_metrics or note:
            data.setdefault("conversion_basis", []).append(
                TranslationEvidence(
                    method="expert",
                    summary=note or "专家人工修改并确认形态目标",
                ).model_dump()
            )
        return MorphTranslationResult(**data)

    # ------------------------------------------------------------------ #
    def _normalize_baseline(self, baseline: dict) -> dict[str, float]:
        self._validate_morph_values(
            baseline,
            source="七项形态基线",
            require_complete=True,
        )
        return {k: float(baseline[k]) for k in MORPH_KEYS}

    @staticmethod
    def _validate_morph_values(
        values: dict,
        source: str,
        require_complete: bool,
    ) -> None:
        if not isinstance(values, dict):
            raise ValueError(f"{source}必须是字典")
        unknown = sorted(set(values) - set(MORPH_KEYS))
        if unknown:
            raise ValueError(f"{source}包含未知指标: {', '.join(unknown)}")
        if require_complete:
            missing = [key for key in MORPH_KEYS if key not in values]
            if missing:
                raise ValueError(f"{source}缺少指标: {', '.join(missing)}")

        ratio_keys = set(MORPH_KEYS) - {"color_richness"}
        for key, raw in values.items():
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{source}.{key}不是有效数值: {raw!r}") from exc
            if not math.isfinite(value):
                raise ValueError(f"{source}.{key}必须是有限数值")
            if key in ratio_keys and not 0.0 <= value <= 1.0:
                raise ValueError(f"{source}.{key}必须位于0到1之间")
            if key == "color_richness" and not 0.0 <= value <= 100.0:
                raise ValueError(f"{source}.{key}必须位于0到100之间")

    def _clamp(self, key: str, value: float) -> float:
        lo, hi = MORPH_BOUNDS[key]
        return float(min(hi, max(lo, value)))

    def _rule_map_targets(
        self,
        targets: dict,
        exp_base: dict,
        morph_base: dict,
    ) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        """依据 mapping_rules.json 的连续系数将体验改善量映射到形态增量。"""
        rules = self.kb.get_mapping_rules().get("rules", [])
        result = copy.deepcopy(morph_base)
        contributions: dict[str, dict[str, float]] = {}

        for rule in rules:
            experience_key = rule.get("experience_key")
            if experience_key not in EXPERIENCE_KEYS:
                continue

            raw_delta = float(targets[experience_key]) - float(exp_base[experience_key])
            direction = rule.get("direction") or EXPERIENCE_DIRECTIONS[experience_key]
            improvement = -raw_delta if direction == "lower_is_better" else raw_delta
            if abs(improvement) < 1e-9:
                continue

            for mk, delta in (rule.get("adjust_per_point") or {}).items():
                if mk in result:
                    applied = float(delta) * improvement
                    result[mk] = result[mk] + applied
                    contributions.setdefault(experience_key, {})[mk] = round(
                        applied, 4
                    )

        return (
            {k: self._clamp(k, result[k]) for k in MORPH_KEYS},
            contributions,
        )

    @staticmethod
    def _format_rule_contributions(
        contributions: dict[str, dict[str, float]],
    ) -> str:
        if not contributions:
            return "七项体感没有变化，规则层保持形态基线"
        summaries = []
        for experience_key in EXPERIENCE_KEYS:
            morph_changes = contributions.get(experience_key)
            if not morph_changes:
                continue
            detail = "、".join(
                f"{MORPH_LABELS_ZH.get(key, key)}{value:+.4f}"
                for key, value in morph_changes.items()
            )
            summaries.append(
                f"{EXPERIENCE_LABELS_ZH.get(experience_key, experience_key)}→{detail}"
            )
        return "；".join(summaries)

    def _blend_with_memory(
        self,
        target: dict,
        morph_base: dict,
        refs: list[dict],
    ) -> dict[str, float]:
        historical_deltas = []
        for ref in refs:
            ref_base = ref.get("baseline_metrics") or {}
            ref_target = ref.get("target_metrics") or {}
            if ref_target:
                historical_deltas.append(
                    {
                        key: float(ref_target[key]) - float(ref_base.get(key, ref_target[key]))
                        for key in MORPH_KEYS
                        if key in ref_target
                    }
                )

        if not historical_deltas:
            return target
        blended = dict(target)
        for k in MORPH_KEYS:
            vals = [delta[k] for delta in historical_deltas if k in delta]
            if vals:
                rag_target = self._clamp(k, float(morph_base[k]) + sum(vals) / len(vals))
                blended[k] = self._clamp(k, 0.75 * target[k] + 0.25 * rag_target)
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
        original_image_path: str,
    ) -> tuple[str, bool]:
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
                "案例: "
                f"id={r.get('id')}; score={r.get('_rag_score')}; "
                f"experience={r.get('experience_targets') or r.get('knobs')}; "
                f"target={r.get('target_metrics')}"
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

        raw = (
            llm_client.chat_with_image(SYSTEM_PROMPT, user, original_image_path)
            if original_image_path
            else llm_client.chat(SYSTEM_PROMPT, user)
        )
        if raw:
            try:
                import json

                text = raw.strip()
                if "```" in text:
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                data = json.loads(text)
                refined: dict[str, float] = {}
                if isinstance(data.get("target_metrics"), dict):
                    for k, v in data["target_metrics"].items():
                        if k in target:
                            refined[k] = self._clamp(k, float(v))
                target.update(refined)
                rationale = str(data.get("rationale") or "").strip()
                return (
                    rationale
                    or self._rule_rationale(exp_base, targets, exp_delta, target),
                    bool(refined),
                )
            except Exception as e:
                print(f"[Translator] LLM JSON 解析失败: {e}")

        return self._rule_rationale(exp_base, targets, exp_delta, target), False

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
    original_image_path: str = "",
) -> MorphTranslationResult:
    return TranslatorAgent().run(
        experience_targets=experience_targets,
        baseline_metrics=baseline_metrics,
        experience_baseline=experience_baseline,
        scene_context=scene_context,
        original_image_path=original_image_path,
    )


if __name__ == "__main__":
    demo = TranslatorAgent().run(
        experience_baseline={key: 3 for key in EXPERIENCE_KEYS},
        experience_targets={
            "comfort": 4,
            "naturalness": 4,
            "safety": 4,
            "relaxation": 4,
            "environmental_disturbance": 2,
            "stay_intention": 4,
            "overall_impression": 4,
        },
        baseline_metrics={"green_view": 0.12, "sky_view": 0.28, "built_ratio": 0.5},
        scene_context="高密度街巷；午后晴天",
    )
    print(demo.model_dump_json(indent=2, ensure_ascii=False))
