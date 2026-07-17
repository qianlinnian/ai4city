"""
================================================================================
制图员 Agent（Cartographer Agent）
文件: agents/cartographer_agent.py
--------------------------------------------------------------------------------
【角色】
  接收原始全景与人工确认后的「形态要素目标值」，结合基线形态、体验目标、
  情景要素、专家建议和历史案例，生成结构化空间布局方案及可被
  World Labs Pano Edit 理解的自然语言修改文本。

【输入】
  - baseline_metrics: dict       原先形态要素（图像解析）
  - target_metrics: dict         确认后的形态要素目标
  - experience_targets: dict     体验目标（可选，辅助措辞）
  - experience_baseline: dict    体验原值（可选，用于RAG检索）
  - scene_context: str           情景要素（可选）
  - original_image_path: str     原始JPG路径（可选，用于多模态LLM）
  - expert_advice: str           专家建议（可选）
  - language: "en" | "zh"        输出语言，默认英文

【输出】
  - ModificationPlan
      .draft_text       可供 World Labs 执行的修改文本
      .object_actions   增加、减少或调整的空间对象及位置、数量
      .spatial_relations / .unchanged_regions / .constraints
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
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    EXPERIENCE_DIRECTIONS,
    EXPERIENCE_KEYS,
    EXPERIENCE_LABELS_ZH,
    MORPH_KEYS,
    MORPH_LABELS_ZH,
    normalize_experience_values,
)
from knowledge_base.kb_store import KnowledgeBase, kb as default_kb
from schemas.models import ModificationPlan, SpatialObjectAction
from utils import llm_client


SYSTEM_PROMPT = (
    "你是城市微空间全景编辑制图员。根据原始全景、形态要素从基线到目标的变化、"
    "情景要素和专家建议，生成结构化空间布局方案以及可被 World Labs Pano Edit 执行的修改文本。"
    "必须明确对象、位置、数量、空间关系、保持不变区域和约束；不得擅自改变建筑体量、道路拓扑或相机视点。"
    "不得擅自删除或移动电力、通信、排水、消防和必要交通标识等基础设施。"
    "只输出 JSON，字段为 plan_summary、object_actions、spatial_relations、unchanged_regions、"
    "constraints、modification_text。object_actions 每项包含 action(add/remove/adjust)、"
    "object_type、position、quantity、attributes、rationale。"
)


class CartographerAgent:
    def __init__(self, knowledge_base: KnowledgeBase | None = None):
        self.kb = knowledge_base or default_kb

    def run(
        self,
        baseline_metrics: dict,
        target_metrics: dict,
        experience_targets: dict | None = None,
        experience_baseline: dict | None = None,
        scene_context: str = "",
        language: str = "en",
        original_image_path: str = "",
        expert_advice: str = "",
    ) -> ModificationPlan:
        exp_base = normalize_experience_values(experience_baseline)
        exp_targets = normalize_experience_values(experience_targets)
        refs = []
        if experience_targets:
            refs = self.kb.retrieve_experience_cases(
                exp_base,
                exp_targets,
                scene_context=scene_context,
                top_k=2,
            )

        templates = [
            (r.get("final_prompt") or r.get("modification_plan") or "")
            for r in refs
            if (r.get("final_prompt") or r.get("modification_plan"))
        ]

        hints = self._build_layout_hints(
            baseline_metrics,
            target_metrics,
            exp_targets,
            exp_base,
        )
        user = self._build_user(
            baseline_metrics,
            target_metrics,
            exp_targets if experience_targets else None,
            scene_context,
            hints,
            templates,
            language,
            expert_advice,
        )
        llm_text = (
            llm_client.chat_with_image(SYSTEM_PROMPT, user, original_image_path)
            if original_image_path
            else llm_client.chat(SYSTEM_PROMPT, user)
        )
        llm_plan = self._parse_llm_plan(llm_text)
        fallback = self._build_structured_layout(
            baseline_metrics,
            target_metrics,
            hints,
            expert_advice=expert_advice,
        )

        actions = fallback["object_actions"]
        if llm_plan and isinstance(llm_plan.get("object_actions"), list):
            llm_actions = self._normalize_llm_actions(llm_plan["object_actions"])
            if llm_actions:
                actions = llm_actions
                actions = self._merge_expert_actions(
                    actions,
                    fallback["object_actions"],
                )

        spatial_relations = self._merge_strings(
            self._string_list((llm_plan or {}).get("spatial_relations")),
            fallback["spatial_relations"],
        )
        unchanged_regions = self._merge_strings(
            self._string_list((llm_plan or {}).get("unchanged_regions")),
            fallback["unchanged_regions"],
        )
        constraints = self._merge_strings(
            self._string_list((llm_plan or {}).get("constraints")),
            fallback["constraints"],
        )
        resolved_layout = {
            **fallback,
            "object_actions": actions,
            "spatial_relations": spatial_relations,
            "unchanged_regions": unchanged_regions,
            "constraints": constraints,
        }

        llm_draft = (
            str(llm_plan.get("modification_text", "")).strip()
            if llm_plan
            else ""
        )
        if llm_draft:
            draft = f"{llm_draft} {self._structured_appendix(resolved_layout, language)}"
        else:
            draft = self._template_plan(
                baseline_metrics,
                target_metrics,
                hints,
                language,
                resolved_layout,
                expert_advice,
            )
        if expert_advice and expert_advice not in draft:
            label = "Expert requirement" if language == "en" else "专家要求"
            draft = f"{draft.rstrip()} {label}: {expert_advice}."

        plan_summary = str(
            (llm_plan or {}).get("plan_summary") or fallback["plan_summary"]
        ).strip()
        if expert_advice and expert_advice not in plan_summary:
            plan_summary += f"；专家建议：{expert_advice}"

        rationale = self._build_rationale(baseline_metrics, target_metrics, exp_targets)

        return ModificationPlan(
            draft_text=draft.strip(),
            language=language,
            plan_summary=plan_summary,
            rationale=rationale,
            layout_hints=hints,
            object_actions=actions,
            spatial_relations=spatial_relations,
            unchanged_regions=unchanged_regions,
            constraints=constraints,
            expert_advice=expert_advice,
            original_image_path=original_image_path,
            rag_references=[ref.get("id", "") for ref in refs if ref.get("id")],
        )

    def apply_human_edit(self, plan: ModificationPlan, human_text: str) -> ModificationPlan:
        """前端人工干预：覆盖自然语言修改方案。"""
        return plan.model_copy(
            update={
                "draft_text": human_text.strip(),
                "rationale": (plan.rationale or "") + " | 人工润色",
            }
        )

    def _build_layout_hints(
        self,
        baseline: dict,
        target: dict,
        knobs: dict,
        experience_baseline: dict,
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
            experience_key = rule.get("experience_key")
            if experience_key not in EXPERIENCE_KEYS:
                continue
            raw_delta = float(knobs[experience_key]) - float(experience_baseline[experience_key])
            direction = rule.get("direction") or EXPERIENCE_DIRECTIONS[experience_key]
            improvement = -raw_delta if direction == "lower_is_better" else raw_delta
            if improvement > 0.25:
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
        expert_advice: str,
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
        if expert_advice:
            lines.append(f"专家建议: {expert_advice}")
        lines.append(
            "请按系统消息要求返回严格 JSON；修改文本应明确空间对象、方位和保持不变区域。"
        )
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
                EXPERIENCE_LABELS_ZH[k]
                for k, v in knobs.items()
                if (
                    EXPERIENCE_DIRECTIONS.get(k) == "higher_is_better"
                    and float(v) >= 4
                )
                or (
                    EXPERIENCE_DIRECTIONS.get(k) == "lower_is_better"
                    and float(v) <= 2
                )
            )
        return (
            f"依据形态目标变化（{', '.join(deltas) or '微调'}）"
            + (f"与体验侧重（{focus}）" if focus else "")
            + "生成结构化空间布局方案与修改文本。"
        )

    @staticmethod
    def _parse_llm_plan(raw: str | None) -> dict | None:
        if not raw:
            return None
        try:
            text = raw.strip()
            if "```" in text:
                text = text.split("```", 2)[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)
            if not isinstance(data, dict):
                return None
            return data
        except (json.JSONDecodeError, TypeError) as exc:
            print(f"[Cartographer] LLM JSON 解析失败，使用规则方案: {exc}")
            return None

    @staticmethod
    def _string_list(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @staticmethod
    def _merge_strings(primary: list[str], required: list[str]) -> list[str]:
        merged: list[str] = []
        for item in [*primary, *required]:
            if item and item not in merged:
                merged.append(item)
        return merged

    @staticmethod
    def _merge_expert_actions(
        primary: list[SpatialObjectAction],
        fallback: list[SpatialObjectAction],
    ) -> list[SpatialObjectAction]:
        merged = list(primary)
        for action in fallback:
            if not action.rationale.startswith("落实专家建议"):
                continue
            if not any(
                existing.object_type == action.object_type
                and existing.position == action.position
                for existing in merged
            ):
                merged.append(action)
        return merged

    @staticmethod
    def _normalize_llm_actions(value: object) -> list[SpatialObjectAction]:
        """兼容国内模型偶尔把 attributes 或 quantity 返回成非字符串类型。"""
        if not isinstance(value, list):
            return []
        normalized: list[SpatialObjectAction] = []
        action_aliases = {
            "add": "add",
            "increase": "add",
            "新增": "add",
            "增加": "add",
            "remove": "remove",
            "delete": "remove",
            "删除": "remove",
            "移除": "remove",
            "adjust": "adjust",
            "modify": "adjust",
            "调整": "adjust",
            "修改": "adjust",
        }
        for item in value:
            if not isinstance(item, dict):
                continue
            data = dict(item)
            action = str(data.get("action", "adjust")).strip().lower()
            data["action"] = action_aliases.get(action, "adjust")
            attributes = data.get("attributes", [])
            if isinstance(attributes, str):
                data["attributes"] = [
                    part.strip()
                    for part in re.split(r"[，,；;]+", attributes)
                    if part.strip()
                ]
            elif not isinstance(attributes, list):
                data["attributes"] = [str(attributes)] if attributes else []
            data["quantity"] = str(data.get("quantity") or "按目标增量适量配置")
            data["rationale"] = str(data.get("rationale") or "")
            try:
                normalized.append(SpatialObjectAction(**data))
            except (TypeError, ValueError) as exc:
                print(f"[Cartographer] 跳过无效 LLM object_action: {exc}")
        return normalized

    @staticmethod
    def _expert_advice_clauses(expert_advice: str) -> tuple[list[str], list[str]]:
        clauses = [
            item.strip(" ：:")
            for item in re.split(r"[，,；;。\n]+", expert_advice or "")
            if item.strip(" ：:")
        ]
        keep_keywords = ("保留", "保持", "不变", "不改", "不得", "避免改变")
        priority_keywords = ("优先", "重点", "着重", "改善", "优化", "增加", "减少", "调整")
        keep = [item for item in clauses if any(key in item for key in keep_keywords)]
        priority = [
            item
            for item in clauses
            if any(key in item for key in priority_keywords) and item not in keep
        ]
        return keep, priority

    @staticmethod
    def _structured_appendix(layout: dict, language: str) -> str:
        actions: list[SpatialObjectAction] = layout["object_actions"]
        if language == "en":
            action_text = "; ".join(
                f"{item.action.upper()} {item.object_type} at {item.position}, "
                f"quantity {item.quantity}, attributes {', '.join(item.attributes)}"
                for item in actions
            )
            return (
                f"Structured execution requirements: {action_text}. "
                f"Spatial relationships: {'; '.join(layout['spatial_relations'])}. "
                f"Keep unchanged: {'; '.join(layout['unchanged_regions'])}. "
                f"Constraints: {'; '.join(layout['constraints'])}."
            )
        labels = {"add": "增加", "remove": "删除", "adjust": "调整"}
        action_text = "；".join(
            f"在{item.position}{labels[item.action]}{item.object_type}，数量为{item.quantity}，"
            f"属性要求为{'、'.join(item.attributes)}"
            for item in actions
        )
        return (
            f"结构化执行要求：{action_text}。"
            f"空间关系：{'；'.join(layout['spatial_relations'])}。"
            f"保持不变区域：{'；'.join(layout['unchanged_regions'])}。"
            f"约束：{'；'.join(layout['constraints'])}。"
        )

    def _build_structured_layout(
        self,
        baseline: dict,
        target: dict,
        hints: list[str],
        expert_advice: str = "",
    ) -> dict:
        """无 LLM 时，根据形态增量生成可审核的对象级空间布局方案。"""
        actions: list[SpatialObjectAction] = []

        green_delta = float(target.get("green_view", 0)) - float(baseline.get("green_view", 0))
        if green_delta > 0.02:
            actions.append(
                SpatialObjectAction(
                    action="add",
                    object_type="乔木、灌木与立面绿化",
                    position="前景边缘、建筑立面及停留节点周边，避开主要视线通廊",
                    quantity=f"分层配置，使绿视率约增加 {green_delta * 100:.1f} 个百分点",
                    attributes=["乡土适生植物", "乔灌草复层", "不遮挡安全视线"],
                    rationale="提升绿视率、自然感与放松感",
                )
            )
        elif green_delta < -0.02:
            actions.append(
                SpatialObjectAction(
                    action="adjust",
                    object_type="过密植被",
                    position="遮挡主要通行视线和出入口的区域",
                    quantity=f"适度疏剪，使绿视率约减少 {abs(green_delta) * 100:.1f} 个百分点",
                    attributes=["保留健康乔木", "优先疏剪灌木下层"],
                    rationale="降低遮挡并提升安全感",
                )
            )

        blue_delta = float(target.get("blue_view", 0)) - float(baseline.get("blue_view", 0))
        if blue_delta > 0.015:
            actions.append(
                SpatialObjectAction(
                    action="add",
                    object_type="小尺度水景或反射性蓝色景观元素",
                    position="不影响通行的中景视觉焦点",
                    quantity=f"控制可视占比增量约 {blue_delta * 100:.1f} 个百分点",
                    attributes=["低维护", "无安全积水风险", "尺度与场地匹配"],
                    rationale="提升蓝视率并形成安静视觉焦点",
                )
            )
        elif blue_delta < -0.015:
            actions.append(
                SpatialObjectAction(
                    action="adjust",
                    object_type="现有水景、蓝色铺装或蓝色视觉元素",
                    position="蓝色占比较高且与场地功能不协调的中前景区域",
                    quantity=f"缩减或替换约 {abs(blue_delta) * 100:.1f} 个百分点的蓝色可视面积",
                    attributes=["优先保留真实水体", "避免把天空误作为可编辑水体"],
                    rationale="降低蓝视率并保持场地真实性",
                )
            )

        sky_delta = float(target.get("sky_view", 0)) - float(baseline.get("sky_view", 0))
        if sky_delta > 0.02:
            actions.append(
                SpatialObjectAction(
                    action="adjust",
                    object_type="上部枝叶、棚架与悬挂遮挡物",
                    position="主要视线通廊上方",
                    quantity=f"局部疏解，使天空可视率约增加 {sky_delta * 100:.1f} 个百分点",
                    attributes=["保留必要遮阴", "不改变建筑轮廓"],
                    rationale="增强开敞度和安全视野",
                )
            )
        elif sky_delta < -0.02:
            actions.append(
                SpatialObjectAction(
                    action="add",
                    object_type="乔木冠层或通透轻型棚架",
                    position="停留节点上方与过度暴露的开敞区域，避开道路和消防通道",
                    quantity=f"控制上部覆盖，使天空可视率约减少 {abs(sky_delta) * 100:.1f} 个百分点",
                    attributes=["保持通透", "不封闭安全视线", "不改变建筑轮廓"],
                    rationale="降低过度暴露感并改善停留体验",
                )
            )

        built_delta = float(target.get("built_ratio", 0)) - float(baseline.get("built_ratio", 0))
        if built_delta < -0.02:
            actions.append(
                SpatialObjectAction(
                    action="adjust",
                    object_type="硬质铺装、围挡与杂乱附属设施",
                    position="连续硬质界面和非必要设施集中区域",
                    quantity=f"软化或整合约 {abs(built_delta) * 100:.1f} 个百分点的可视人造界面",
                    attributes=["透水铺装", "自然材质", "设施统一收纳"],
                    rationale="降低人造物压迫与环境干扰",
                )
            )
        elif built_delta > 0.02:
            actions.append(
                SpatialObjectAction(
                    action="add",
                    object_type="低矮街道家具、连续铺装边界或必要服务设施",
                    position="现有停留节点和功能缺失的场地边缘，不占用主要通道",
                    quantity=f"以小尺度对象补充约 {built_delta * 100:.1f} 个百分点的可视人造界面",
                    attributes=["不新增建筑体量", "尺度克制", "功能明确"],
                    rationale="在不改变建筑和道路结构的前提下满足人造物占比目标",
                )
            )

        edge_delta = float(target.get("edge_density", 0)) - float(
            baseline.get("edge_density", 0)
        )
        if edge_delta < -0.01:
            actions.append(
                SpatialObjectAction(
                    action="remove",
                    object_type="重复标识、零散小设施与杂乱边界",
                    position="道路边缘、墙面和视线焦点周边",
                    quantity="仅移除非必要对象，保留功能与安全标识",
                    attributes=["边界连续", "视觉语言统一"],
                    rationale="降低边缘密度和视觉噪声",
                )
            )
        elif edge_delta > 0.01:
            actions.append(
                SpatialObjectAction(
                    action="adjust",
                    object_type="铺装分区、绿化边界与导向性线性元素",
                    position="路径转折、功能分区交界和需要强化识别的入口",
                    quantity=f"有序强化边界，使边缘密度约增加 {edge_delta * 100:.1f} 个百分点",
                    attributes=["连续清晰", "避免零碎装饰", "服务空间识别"],
                    rationale="增加可读的空间边界，而非制造视觉杂乱",
                )
            )

        color_delta = float(target.get("color_richness", 0)) - float(baseline.get("color_richness", 0))
        if color_delta > 0.5:
            actions.append(
                SpatialObjectAction(
                    action="add",
                    object_type="季相花卉与低饱和度材质点缀",
                    position="入口、停留节点和中景绿化带",
                    quantity=f"增加约 {color_delta:.1f} 个有效色彩层次",
                    attributes=["控制色彩面积", "避免高饱和杂乱"],
                    rationale="提高色彩丰富度和总体感",
                )
            )
        elif color_delta < -0.5:
            actions.append(
                SpatialObjectAction(
                    action="adjust",
                    object_type="高饱和标识与零散装饰色",
                    position="全景中视觉冲突明显的界面",
                    quantity="统一主辅色并减少非必要装饰色",
                    attributes=["低饱和", "材质协调"],
                    rationale="减少环境干扰并增强整体协调",
                )
            )

        skyline_delta = float(target.get("skyline_variance", 0)) - float(
            baseline.get("skyline_variance", 0)
        )
        if abs(skyline_delta) > 0.003:
            actions.append(
                SpatialObjectAction(
                    action="adjust",
                    object_type="乔木冠层与轻型景观构筑物高度",
                    position="建筑天际线前方的景观层",
                    quantity="通过高低错落微调轮廓，不新增或改造建筑体量",
                    attributes=["保持建筑天际线", "避免遮挡地标"],
                    rationale="匹配目标天际线变化率",
                )
            )

        keep_clauses, priority_clauses = self._expert_advice_clauses(expert_advice)
        for clause in priority_clauses:
            actions.append(
                SpatialObjectAction(
                    action="adjust",
                    object_type="专家指定重点区域的现有空间对象",
                    position=clause,
                    quantity="在七项形态目标范围内优先配置资源和修改强度",
                    attributes=["服从专家建议", "保持功能与安全约束"],
                    rationale=f"落实专家建议：{clause}",
                )
            )

        if not actions:
            actions.append(
                SpatialObjectAction(
                    action="adjust",
                    object_type="现有绿化、铺装与街道家具",
                    position="原位置微调",
                    quantity="轻量优化，不新增大体量对象",
                    attributes=["保持空间结构", "保持功能"],
                    rationale="形态目标变化较小，以保守优化为主",
                )
            )

        unchanged_regions = [
            "原始相机视点与360°等距柱状投影视角",
            "建筑主体体量、立面开口及道路拓扑",
            "主要出入口、消防通道和必要安全设施",
            "未经专家确认的电力、通信、排水等市政基础设施",
        ]
        unchanged_regions.extend(f"专家指定保持不变：{item}" for item in keep_clauses)

        constraints = [
            "所有新增对象应符合真实尺度、透视、光照和材质逻辑",
            "保持全景左右接缝连续，不产生重复对象或断裂边缘",
            "不新增未经专家确认的建筑，不改变道路与场地基本功能",
            "不得擅自删除或移动电线、管线、消防设施和必要交通标识",
            "优先使用低维护、适生且不遮挡安全视线的景观对象",
        ]
        if expert_advice:
            constraints.append(f"完整执行专家建议：{expert_advice}")

        summary = "；".join(hints[:4])
        if expert_advice:
            summary += f"；专家建议：{expert_advice}"

        return {
            "plan_summary": summary,
            "object_actions": actions,
            "spatial_relations": [
                "新增对象不得占用主要步行轴线、消防通道和出入口",
                "乔木形成上层遮阴，灌木限定边界，座椅面向开敞且可观察区域",
                "水景、花卉和街道家具作为中景节点，避免在全景接缝处形成突变",
            ],
            "unchanged_regions": unchanged_regions,
            "constraints": constraints,
        }

    def _template_plan(
        self,
        baseline: dict,
        target: dict,
        hints: list[str],
        language: str,
        layout: dict,
        expert_advice: str,
    ) -> str:
        if language == "zh":
            return self._zh_plan(
                baseline, target, hints, layout, expert_advice
            )
        return self._en_plan(
            baseline, target, hints, layout, expert_advice
        )

    def _en_plan(
        self,
        baseline: dict,
        target: dict,
        hints: list[str],
        layout: dict,
        expert_advice: str,
    ) -> str:
        parts = [
            "Edit this equirectangular 360-degree urban micro-space panorama "
            "while preserving overall geometry and camera viewpoint."
        ]
        if expert_advice:
            parts.append(f"Expert requirement: {expert_advice}.")
        parts.append("Object-level edits:")
        for action in layout["object_actions"]:
            parts.append(
                f"{action.action.upper()} {action.object_type} at {action.position}; "
                f"quantity: {action.quantity}; attributes: {', '.join(action.attributes)}."
            )
        parts.append(
            "Spatial relationships: " + "; ".join(layout["spatial_relations"]) + "."
        )
        parts.append(
            "Keep unchanged: " + "; ".join(layout["unchanged_regions"]) + "."
        )
        parts.append("Constraints: " + "; ".join(layout["constraints"]) + ".")
        metric_targets = []
        for key in MORPH_KEYS:
            value = float(target[key])
            label = MORPH_LABELS_ZH.get(key, key)
            metric_targets.append(
                f"{label}={value:.2f}"
                if key == "color_richness"
                else f"{label}={value * 100:.1f}%"
            )
        parts.append("Target morphology: " + ", ".join(metric_targets) + ".")
        parts.append(
            "Maintain photorealistic lighting, clean edges, and physically plausible materials. "
            "Keep the panorama seam continuous and do not invent new buildings or alter road topology."
        )
        return " ".join(parts)

    def _zh_plan(
        self,
        baseline: dict,
        target: dict,
        hints: list[str],
        layout: dict,
        expert_advice: str,
    ) -> str:
        parts = [
            "请在保持原有360°全景几何结构与视点不变的前提下，"
            "对高密度城市微空间进行局部优化编辑。"
        ]
        if expert_advice:
            parts.append(f"专家要求：{expert_advice}。")
        action_labels = {"add": "增加", "remove": "删除", "adjust": "调整"}
        parts.append("对象级修改：")
        for action in layout["object_actions"]:
            parts.append(
                f"在{action.position}{action_labels[action.action]}{action.object_type}，"
                f"数量为{action.quantity}，属性要求为{'、'.join(action.attributes)}。"
            )
        parts.append("空间关系：" + "；".join(layout["spatial_relations"]) + "。")
        parts.append("保持不变区域：" + "；".join(layout["unchanged_regions"]) + "。")
        parts.append("约束：" + "；".join(layout["constraints"]) + "。")
        named = []
        for k in MORPH_KEYS:
            v = target[k]
            label = MORPH_LABELS_ZH.get(k, k)
            if k == "color_richness":
                named.append(f"{label}约{float(v):.1f}")
            else:
                named.append(f"{label}约{float(v)*100:.0f}%")
        parts.append("目标形态指标：" + "、".join(named) + "。")
        parts.append(
            "保持光影真实、尺度准确、材质物理合理，并确保全景左右接缝连续。"
        )
        return "".join(parts)


def run_cartographer(
    baseline_metrics: dict,
    target_metrics: dict,
    experience_targets: dict | None = None,
    experience_baseline: dict | None = None,
    language: str = "en",
    original_image_path: str = "",
    expert_advice: str = "",
) -> ModificationPlan:
    return CartographerAgent().run(
        baseline_metrics=baseline_metrics,
        target_metrics=target_metrics,
        experience_targets=experience_targets,
        experience_baseline=experience_baseline,
        language=language,
        original_image_path=original_image_path,
        expert_advice=expert_advice,
    )


if __name__ == "__main__":
    plan = CartographerAgent().run(
        baseline_metrics={"green_view": 0.12, "sky_view": 0.3, "built_ratio": 0.5},
        target_metrics={"green_view": 0.35, "sky_view": 0.25, "built_ratio": 0.4},
        experience_targets={
            "comfort": 4,
            "naturalness": 4,
            "safety": 4,
            "relaxation": 4,
            "environmental_disturbance": 2,
            "stay_intention": 4,
            "overall_impression": 4,
        },
    )
    print(plan.draft_text)
