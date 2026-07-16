"""
================================================================================
制图员 Agent（Cartographer Agent）
文件: agents/cartographer_agent.py
--------------------------------------------------------------------------------
【角色】
  接收人工确认后的「形态要素目标值」，结合基线形态、体验目标与情景要素，
  生成可被文生图模型（World Labs Pano Edit）理解的自然语言修改方案。

【输入】
  - baseline_metrics: dict       原先形态要素（图像解析）
  - target_metrics: dict         确认后的形态要素目标
  - experience_targets: dict     体验目标（可选，辅助措辞）
  - scene_context: str           情景要素（可选）
  - language: "en" | "zh"        输出语言，默认英文

【输出】
  - ModificationPlan
      .draft_text       200-300 字自然语言修改方案（如「在右侧墙面增加一棵乔木…」）
      .language
      .rationale        简要理由
      .layout_hints     结构化修改要点

【输出到哪里】
  → 前端展示，供人工干预修改自然语言方案
  → 确认后传给 worldlabs_agent.run（文生图工具）

【怎么调用】
  from agents.cartographer_agent import CartographerAgent
  agent = CartographerAgent()
  plan = agent.run(
      baseline_metrics={...},
      target_metrics={...},
      experience_targets={...},
  )
  final = agent.apply_human_edit(plan, human_text="...")
================================================================================
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import EXPERIENCE_LABELS_ZH, MORPH_KEYS, MORPH_LABELS_ZH
from knowledge_base.kb_store import KnowledgeBase, kb as default_kb
from schemas.models import ModificationPlan
from utils import llm_client


SYSTEM_PROMPT = (
    "你是城市微空间全景编辑制图员。根据形态要素从基线到目标的变化，"
    "写出一段精准、富有画面感的自然语言修改方案（可被文生图模型执行），"
    "包含方位、植物、材质、光影。只输出修改方案正文，不要解释。"
)


class CartographerAgent:
    def __init__(self, knowledge_base: KnowledgeBase | None = None):
        self.kb = knowledge_base or default_kb

    def run(
        self,
        baseline_metrics: dict,
        target_metrics: dict,
        experience_targets: dict | None = None,
        scene_context: str = "",
        language: str = "en",
    ) -> ModificationPlan:
        refs = []
        if experience_targets:
            refs = self.kb.retrieve_similar(experience_targets, top_k=2)

        templates = [
            (r.get("final_prompt") or r.get("modification_plan") or "")
            for r in refs
            if (r.get("final_prompt") or r.get("modification_plan"))
        ]

        hints = self._build_layout_hints(baseline_metrics, target_metrics, experience_targets or {})
        user = self._build_user(
            baseline_metrics,
            target_metrics,
            experience_targets,
            scene_context,
            hints,
            templates,
            language,
        )
        llm_text = llm_client.chat(SYSTEM_PROMPT, user)
        draft = llm_text or self._template_plan(
            baseline_metrics, target_metrics, hints, language
        )

        rationale = self._build_rationale(baseline_metrics, target_metrics, experience_targets)

        return ModificationPlan(
            draft_text=draft.strip(),
            language=language,
            rationale=rationale,
            layout_hints=hints,
        )

    def apply_human_edit(self, plan: ModificationPlan, human_text: str) -> ModificationPlan:
        """前端人工干预：覆盖自然语言修改方案。"""
        return ModificationPlan(
            draft_text=human_text.strip(),
            language=plan.language,
            rationale=(plan.rationale or "") + " | 人工润色",
            layout_hints=plan.layout_hints,
        )

    def _build_layout_hints(
        self,
        baseline: dict,
        target: dict,
        knobs: dict,
    ) -> list[str]:
        hints: list[str] = []
        rules = self.kb.get_mapping_rules().get("rules", [])

        for k in MORPH_KEYS:
            b = float(baseline.get(k, 0))
            t = float(target.get(k, 0))
            delta = t - b
            if abs(delta) < 0.02 and k != "color_richness":
                continue
            if k == "color_richness" and abs(delta) < 0.5:
                continue
            label = MORPH_LABELS_ZH.get(k, k)
            if delta > 0:
                hints.append(f"提升{label}（{b:.2f}→{t:.2f}）")
            else:
                hints.append(f"降低{label}（{b:.2f}→{t:.2f}）")

        for rule in rules:
            high_key = rule.get("when_high")
            if high_key and float(knobs.get(high_key, 3)) >= 4:
                hint = rule.get("layout_hint")
                if hint and hint not in hints:
                    hints.append(hint)

        if not hints:
            hints.append("轻度优化绿化与界面材质，保持空间结构不变")
        return hints

    def _build_user(
        self,
        baseline: dict,
        target: dict,
        knobs: dict | None,
        scene_context: str,
        hints: list[str],
        templates: list[str],
        language: str,
    ) -> str:
        lines = [
            f"输出语言: {'English' if language == 'en' else '中文'}",
            f"形态基线: {json.dumps(baseline, ensure_ascii=False)}",
            f"形态目标: {json.dumps(target, ensure_ascii=False)}",
            f"修改要点: {json.dumps(hints, ensure_ascii=False)}",
        ]
        if knobs:
            knob_desc = ", ".join(
                f"{EXPERIENCE_LABELS_ZH.get(k, k)}={v}" for k, v in knobs.items()
            )
            lines.append(f"体验目标: {knob_desc}")
        if scene_context:
            lines.append(f"情景要素: {scene_context}")
        if templates:
            lines.append(f"优质历史方案参考: {templates[0][:400]}")
        lines.append("请生成 200-300 字的全景编辑自然语言方案。")
        return "\n".join(lines)

    def _build_rationale(
        self,
        baseline: dict,
        target: dict,
        knobs: dict | None,
    ) -> str:
        deltas = []
        for k in MORPH_KEYS:
            b, t = float(baseline.get(k, 0)), float(target.get(k, 0))
            if abs(t - b) > 0.01:
                deltas.append(f"{MORPH_LABELS_ZH.get(k, k)} {b:.2f}→{t:.2f}")
        focus = ""
        if knobs:
            focus = "、".join(
                EXPERIENCE_LABELS_ZH[k] for k, v in knobs.items() if float(v) >= 4
            )
        return (
            f"依据形态目标变化（{', '.join(deltas) or '微调'}）"
            + (f"与体验侧重（{focus}）" if focus else "")
            + "生成文生图修改方案。"
        )

    def _template_plan(
        self,
        baseline: dict,
        target: dict,
        hints: list[str],
        language: str,
    ) -> str:
        if language == "zh":
            return self._zh_plan(baseline, target, hints)
        return self._en_plan(baseline, target, hints)

    def _en_plan(self, baseline: dict, target: dict, hints: list[str]) -> str:
        parts = [
            "Edit this equirectangular 360-degree urban micro-space panorama "
            "while preserving overall geometry and camera viewpoint."
        ]
        for h in hints:
            parts.append(f"Apply: {h}.")
        gvf = target.get("green_view")
        if gvf is not None:
            base = baseline.get("green_view")
            if base is not None and float(gvf) > float(base) + 0.03:
                parts.append(
                    f"Increase green view from about {float(base)*100:.0f}% toward "
                    f"{float(gvf)*100:.0f}% with layered shrubs and climbing plants on walls."
                )
        if float(target.get("sky_view", 0)) > float(baseline.get("sky_view", 0)) + 0.02:
            parts.append(
                "Open up the upper view toward the sky, trim overhanging branches, "
                "and keep sightlines clear for safety."
            )
        if float(target.get("built_ratio", 1)) < float(baseline.get("built_ratio", 0)) - 0.03:
            parts.append(
                "Soften hard paved surfaces with permeable materials and add wooden seating edges."
            )
        if float(target.get("color_richness", 0)) > float(baseline.get("color_richness", 0)) + 0.5:
            parts.append(
                "Add seasonal flowering shrubs in the mid-ground for richer color layers."
            )
        parts.append(
            "Maintain photorealistic lighting, clean edges, and physically plausible materials. "
            "Do not invent new buildings or change road topology drastically."
        )
        return " ".join(parts)

    def _zh_plan(self, baseline: dict, target: dict, hints: list[str]) -> str:
        parts = [
            "请在保持原有360°全景几何结构与视点不变的前提下，"
            "对高密度城市微空间进行局部优化编辑。"
        ]
        for h in hints:
            parts.append(f"{h}。")
        named = []
        for k, v in target.items():
            label = MORPH_LABELS_ZH.get(k, k)
            if k == "color_richness":
                named.append(f"{label}约{float(v):.1f}")
            else:
                named.append(f"{label}约{float(v)*100:.0f}%")
        parts.append("目标形态指标：" + "、".join(named) + "。")
        parts.append(
            "保持光影真实、材质物理合理，避免大幅改变建筑体量与道路走向，"
            "营造安静可停留的亲自然氛围。"
        )
        return "".join(parts)


def run_cartographer(
    baseline_metrics: dict,
    target_metrics: dict,
    experience_targets: dict | None = None,
    language: str = "en",
) -> ModificationPlan:
    return CartographerAgent().run(
        baseline_metrics, target_metrics, experience_targets, language=language
    )


if __name__ == "__main__":
    plan = CartographerAgent().run(
        baseline_metrics={"green_view": 0.12, "sky_view": 0.3, "built_ratio": 0.5},
        target_metrics={"green_view": 0.35, "sky_view": 0.25, "built_ratio": 0.4},
        experience_targets={"comfort": 4, "restoration": 5, "safety": 3, "pleasure": 4, "stay": 4},
    )
    print(plan.draft_text)
