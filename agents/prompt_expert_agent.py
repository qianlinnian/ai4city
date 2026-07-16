"""
@deprecated 提示词专家 Agent 已合并至制图员 Agent（cartographer_agent.py）。

v2 流程中，制图员直接输出自然语言修改方案，不再需要单独的提示词专家。
保留此文件仅为向后兼容，请使用 CartographerAgent。
"""

from __future__ import annotations

import warnings

from agents.cartographer_agent import CartographerAgent
from schemas.models import ModificationPlan, PromptDraft


class PromptExpertAgent:
    """@deprecated 请使用 CartographerAgent"""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "PromptExpertAgent 已废弃，请使用 CartographerAgent",
            DeprecationWarning,
            stacklevel=2,
        )
        self._cartographer = CartographerAgent(*args, **kwargs)

    def run(
        self,
        target_metrics: dict,
        layout_actions: list | None = None,
        baseline_metrics: dict | None = None,
        experience_targets: dict | None = None,
        language: str = "en",
    ) -> PromptDraft:
        plan = self._cartographer.run(
            baseline_metrics=baseline_metrics or {},
            target_metrics=target_metrics,
            experience_targets=experience_targets,
            language=language,
        )
        return PromptDraft(
            draft_prompt=plan.draft_text,
            language=plan.language,
            layout_actions=layout_actions or [],
        )

    def apply_human_edit(self, draft: PromptDraft, human_prompt: str) -> PromptDraft:
        return PromptDraft(
            draft_prompt=human_prompt.strip(),
            language=draft.language,
            layout_actions=draft.layout_actions,
        )


def run_prompt_expert(*args, **kwargs) -> PromptDraft:
    return PromptExpertAgent().run(*args, **kwargs)
