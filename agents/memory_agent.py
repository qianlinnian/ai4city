"""
================================================================================
记忆 Agent（Memory Agent）
文件: agents/memory_agent.py
--------------------------------------------------------------------------------
【角色】
  将全流程关键数据（体验、形态、修改方案、质检结果、多人体验）写入本地知识库，
  供翻译官 / 制图员后续 Few-shot 检索。

【输入】
  - knobs / experience_targets: dict        体验目标
  - experience_baseline: dict               体验原值
  - baseline_metrics / target_metrics: dict 形态基线与目标
  - scene_context: str                      情景要素
  - modification_plan: str                  制图员自然语言方案
  - final_prompt: str                       人工确认后的方案
  - measured_after: dict                    质检实测
  - human_corrected_metrics: dict           人工纠偏指标
  - pre_edit_experience: list               修改前多人体验
  - post_edit_experience: list              修改后多人体验
  - score, notes

【输出】
  - MemoryRecord → knowledge_base/data/memories.json

【怎么调用】
  from agents.memory_agent import MemoryAgent
  mem = MemoryAgent().store_feedback(...)
================================================================================
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from knowledge_base.kb_store import KnowledgeBase, kb as default_kb
from schemas.models import MemoryRecord
from utils import llm_client


class MemoryAgent:
    def __init__(self, knowledge_base: KnowledgeBase | None = None):
        self.kb = knowledge_base or default_kb

    def summarize_diff(self, draft_plan: str, final_plan: str) -> str:
        if draft_plan.strip() == final_plan.strip():
            return "人工未修改修改方案。"
        system = "对比 AI 制图员方案与人工定稿，用一两句话总结偏好差异。"
        user = f"草稿:\n{draft_plan}\n\n定稿:\n{final_plan}"
        llm = llm_client.chat(system, user)
        if llm:
            return llm.strip()
        d_len, f_len = len(draft_plan), len(final_plan)
        if f_len > d_len + 40:
            return "人工补充了更多材质/方位/植物细节"
        if f_len + 40 < d_len:
            return "人工精简了修改方案"
        return "人工对措辞与细节进行了润色。"

    def store_feedback(
        self,
        knobs: dict,
        modification_plan: str = "",
        final_prompt: str = "",
        baseline_metrics: dict | None = None,
        target_metrics: dict | None = None,
        experience_baseline: dict | None = None,
        scene_context: str = "",
        measured_after: dict | None = None,
        human_corrected_metrics: dict | None = None,
        pre_edit_experience: list | None = None,
        post_edit_experience: list | None = None,
        score: float | None = None,
        notes: str = "",
        # 兼容旧字段
        intent: str = "",
        layout: list | None = None,
        draft_prompt: str = "",
    ) -> MemoryRecord:
        draft = modification_plan or draft_prompt
        final = final_prompt
        diff_summary = self.summarize_diff(draft, final)

        if human_corrected_metrics and measured_after:
            metric_diffs = []
            for k, hv in human_corrected_metrics.items():
                mv = measured_after.get(k)
                if mv is not None and abs(float(hv) - float(mv)) > 1e-6:
                    metric_diffs.append(f"{k}: 实测{mv}→人工认定{hv}")
            if metric_diffs:
                diff_summary += " | 指标纠偏: " + ", ".join(metric_diffs)

        record = MemoryRecord(
            id=str(uuid.uuid4()),
            knobs=knobs,
            experience_baseline=experience_baseline or {},
            baseline_metrics=baseline_metrics or {},
            target_metrics=target_metrics or {},
            scene_context=scene_context,
            modification_plan=draft,
            final_prompt=final,
            measured_after=measured_after or {},
            human_corrected_metrics=human_corrected_metrics or {},
            pre_edit_experience=pre_edit_experience or [],
            post_edit_experience=post_edit_experience or [],
            diff_summary=diff_summary,
            score=score,
            notes=notes,
        )
        return self.kb.add_memory(record)


def run_memory_store(**kwargs) -> MemoryRecord:
    return MemoryAgent().store_feedback(**kwargs)
