"""Task 2/3 共用的全景多图场景理解 Agent。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import SCENE_UNDERSTANDING_ENABLED
from schemas.models import PanoramaSceneInventory, PanoramaViewSet, SceneElement
from utils import llm_client
from utils.panorama_views import PanoramaViewGenerator


SCENE_SYSTEM_PROMPT = """你是城市微空间全景场景核查员。输入是一张完整全景概览和多个带 yaw、pitch、FOV 标记的普通透视图。
你的任务只做可核验的场景清单，不提出设计方案，不推测被遮挡对象。
特别规则：蓝天、蓝色招牌、蓝色铺装和反光不得当作真实水体；不能确认时写入 ambiguities，water 保持为空。每个对象必须给出 evidence_view_ids 和 0~1 confidence，并尽量提供 yaw_range、方位或图像区域。
只输出 JSON，不要 Markdown。字段必须为 status、roads、buildings、entrances、vegetation、water、street_furniture、infrastructure、editable_objects、fixed_regions、spatial_relations、panorama_seam_constraints、ambiguities、confidence、evidence_view_ids。
前九个对象类别中的每项字段为 name、description、position、quantity、yaw_range、pitch、image_region、confidence、evidence_view_ids。不得把不确定对象写成已确认事实。"""


_ELEMENT_FIELDS = (
    "roads",
    "buildings",
    "entrances",
    "vegetation",
    "water",
    "street_furniture",
    "infrastructure",
    "editable_objects",
    "fixed_regions",
)


class SceneUnderstandingAgent:
    def __init__(
        self,
        view_generator: PanoramaViewGenerator | None = None,
        *,
        enabled: bool = SCENE_UNDERSTANDING_ENABLED,
    ) -> None:
        self.view_generator = view_generator or PanoramaViewGenerator()
        self.enabled = enabled

    @staticmethod
    def _degraded(
        image_path: str | Path,
        reason: str,
        view_set: PanoramaViewSet | None = None,
        *,
        status: str = "degraded",
    ) -> PanoramaSceneInventory:
        return PanoramaSceneInventory(
            status=status,
            source_image_path=str(Path(image_path).resolve()),
            view_metadata=view_set.views if view_set else [],
            panorama_seam_constraints=(
                ["保持全景左右接缝连续，接缝附近对象不得重复或断裂"]
                if view_set
                else []
            ),
            confidence=0.0,
            degradation_reason=reason,
        )

    def run(
        self,
        image_path: str | Path,
        *,
        image_id: str = "",
    ) -> PanoramaSceneInventory:
        if not self.enabled:
            return self._degraded(
                image_path,
                "场景理解已由配置关闭",
                status="disabled",
            )
        try:
            view_set = self.view_generator.generate(
                image_path,
                source_image_id=image_id,
            )
        except Exception as exc:
            return self._degraded(image_path, f"全景视图生成失败: {exc}")

        view_manifest = [
            {
                "view_id": view.view_id,
                "yaw": view.yaw,
                "pitch": view.pitch,
                "fov": view.fov,
                "is_overview": view.is_overview,
            }
            for view in view_set.views
        ]
        user_prompt = (
            "请先逐视图观察，再合并成一份结构化场景清单。仅依据画面证据；"
            "同一对象跨视图出现时去重，并记录所有证据视图。\n"
            + json.dumps({"view_manifest": view_manifest}, ensure_ascii=False)
        )
        raw = llm_client.chat_with_images(
            SCENE_SYSTEM_PROMPT,
            user_prompt,
            view_set.views,
            temperature=0.1,
        )
        if not raw:
            return self._degraded(
                image_path,
                "多模态模型未返回场景清单；Task 2/3 将在空场景上下文下继续",
                view_set,
            )
        try:
            text = raw.strip()
            if "```" in text:
                text = text.split("```", 2)[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError("场景清单必须是 JSON 对象")
            data.update(
                {
                    "status": "ok",
                    "source_image_path": str(Path(image_path).resolve()),
                    "view_metadata": [view.model_dump() for view in view_set.views],
                    "degradation_reason": "",
                }
            )
            inventory = PanoramaSceneInventory(**data)
            return self._filter_unverifiable(inventory)
        except Exception as exc:
            return self._degraded(
                image_path,
                f"场景清单 JSON 无效: {exc}",
                view_set,
            )

    @staticmethod
    def _filter_unverifiable(
        inventory: PanoramaSceneInventory,
    ) -> PanoramaSceneInventory:
        """丢弃无有效视图证据的对象，尤其避免凭空补充水体。"""
        valid_ids = {view.view_id for view in inventory.view_metadata}
        updates: dict[str, Any] = {}
        ambiguities = list(inventory.ambiguities)
        for field_name in _ELEMENT_FIELDS:
            accepted: list[SceneElement] = []
            for element in getattr(inventory, field_name):
                evidence = [item for item in element.evidence_view_ids if item in valid_ids]
                minimum_confidence = 0.5 if field_name == "water" else 0.25
                if not evidence or element.confidence < minimum_confidence:
                    ambiguities.append(
                        f"未作为已确认{field_name}采用：{element.name}（证据不足）"
                    )
                    continue
                accepted.append(element.model_copy(update={"evidence_view_ids": evidence}))
            updates[field_name] = accepted
        updates["ambiguities"] = list(dict.fromkeys(ambiguities))
        updates["evidence_view_ids"] = [
            item for item in inventory.evidence_view_ids if item in valid_ids
        ]
        return inventory.model_copy(update=updates)


__all__ = ["SCENE_SYSTEM_PROMPT", "SceneUnderstandingAgent"]
