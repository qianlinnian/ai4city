"""RAG 扩展接口。

当前阶段默认不执行检索。后续准备好课本、案例或专家知识后，可实现
``TranslationRagProvider.retrieve`` 并注入 ``TranslatorAgent``，无需改变
翻译官的输入输出协议。
"""

from __future__ import annotations

from typing import Any, Protocol


class TranslationRagProvider(Protocol):
    """翻译官可选知识检索接口。"""

    def retrieve(
        self,
        *,
        experience_records: list[dict[str, Any]],
        experience_targets: dict[str, float],
        baseline_metrics: dict[str, float],
        scene_context: str,
    ) -> list[dict[str, Any]]:
        """返回可注入 Prompt 的知识条目；当前项目暂不提供实现。"""


class LayoutRagProvider(Protocol):
    """Task 3 空间布局可选知识检索接口。"""

    def retrieve(
        self,
        *,
        baseline_metrics: dict[str, float],
        target_metrics: dict[str, float],
        scene_context: str,
        expert_advice: str,
    ) -> list[dict[str, Any]]:
        """返回可注入布局Prompt的知识条目；当前项目暂不提供实现。"""
