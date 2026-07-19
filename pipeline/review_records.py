"""Markdown audit records for scene-specific renovation-plan rounds."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def review_record_path(
    review_dir: str | Path, *, image_name: str, session_id: str
) -> Path:
    """Return a stable, repository-local Markdown path for one session."""
    stem = Path(image_name or "scene").stem or "scene"
    safe_stem = _SAFE_NAME.sub("_", stem).strip("._") or "scene"
    return Path(review_dir).resolve() / f"{safe_stem}_{session_id[:8]}_review.md"


def _json_block(value: Any) -> str:
    return json.dumps(value or {}, ensure_ascii=False, indent=2, default=str)


def _metrics_lines(values: Mapping[str, Any] | None) -> str:
    if not values:
        return "- 无"
    return "\n".join(f"- `{key}`: {value}" for key, value in values.items())


def render_review_round(state: Mapping[str, Any], *, event: str) -> str:
    """Render one self-contained plan-review round without altering state."""
    plan = dict(state.get("modification_plan") or {})
    translation = dict(state.get("morph_translation") or {})
    actions = plan.get("object_actions") or []
    action_lines = []
    for index, action in enumerate(actions, start=1):
        if not isinstance(action, Mapping):
            continue
        action_lines.append(
            f"{index}. `{action.get('action', 'adjust')}` "
            f"{action.get('object_type', '')}｜位置：{action.get('position', '')}｜"
            f"数量/范围：{action.get('quantity', '')}"
        )

    references = plan.get("rag_references") or translation.get("references_used") or []
    final_prompt = str(state.get("final_prompt") or plan.get("draft_text") or "")
    unchanged_regions = plan.get("unchanged_regions") or []
    timestamp = datetime.now().isoformat(timespec="seconds")
    return "\n\n".join(
        [
            f"## 轮次 {state.get('translator_round') or 1}：{event}",
            f"- 记录时间：{timestamp}\n"
            f"- 会话：`{state.get('session_id', '')}`\n"
            f"- 阶段：`{state.get('stage', '')}`\n"
            f"- 场景模板：`{state.get('cartographer_scene_profile') or '未确认'}`\n"
            f"- 原图：`{state.get('image_path', '')}`\n"
            f"- 专家意见：{state.get('expert_morph_note') or '无'}",
            "### 情景要素\n" + str(state.get("scene_context_text") or "无"),
            "### 场景清单（图片理解结果）\n```json\n"
            + _json_block(state.get("scene_understanding"))
            + "\n```",
            "### 七项形态指标\n"
            + "#### 修改前\n"
            + _metrics_lines(state.get("baseline_metrics"))
            + "\n\n#### 确认目标\n"
            + _metrics_lines(state.get("confirmed_target_metrics")),
            "### 方案摘要\n" + str(plan.get("plan_summary") or "无"),
            "### 对象级修改\n" + ("\n".join(action_lines) or "- 无"),
            "### 保持不变区域\n"
            + ("\n".join(f"- {item}" for item in unchanged_regions) or "- 无"),
            "### RAG 参考条目\n"
            + ("\n".join(f"- `{item}`" for item in references) or "- 无"),
            "### 最终执行空间布局方案（原样发送给 Seedream）\n```text\n"
            + final_prompt
            + "\n```",
        ]
    )


def append_review_round(
    state: Mapping[str, Any], *, event: str, review_dir: str | Path
) -> Path:
    """Append a plan-review round to the session's Markdown record."""
    destination = review_record_path(
        review_dir,
        image_name=str(state.get("image_name") or "scene"),
        session_id=str(state.get("session_id") or "unknown"),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = destination.read_text(encoding="utf-8")
    else:
        existing = (
            "# 场景改造方案复核记录\n\n"
            f"- 图片：`{state.get('image_name', '')}`\n"
            f"- 会话：`{state.get('session_id', '')}`\n"
        )
    destination.write_text(
        existing.rstrip() + "\n\n" + render_review_round(state, event=event) + "\n",
        encoding="utf-8",
    )
    return destination
