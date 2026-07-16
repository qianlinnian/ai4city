"""
================================================================================
学习 Agent（Learning Agent）
文件: agents/learning_agent.py
--------------------------------------------------------------------------------
【角色】
  学习每次「体验要素 → 形态要素」翻译是否准确。对比翻译官预测目标与
  人工修正后的形态目标，积累反馈供翻译官后续决策参考。

  ⚠️ 当前为占位实现：接口已预留，不影响主流程运行。
  即使不启用学习 Agent，全流程仍可正常执行。

【输入】
  - record_translation_feedback(...)  每次人工确认形态目标后调用
      experience_baseline, experience_targets
      predicted_target_metrics   翻译官给出的目标
      human_corrected_metrics    人工修正后的目标
      accurate: bool             人工认定是否准确（可选）
      notes: str

  - get_morph_correction(...)   翻译官运行时可选查询
      experience_baseline, experience_targets
      baseline_metrics, predicted_target

【输出】
  - get_morph_correction → dict | None   对形态目标的修正建议（无数据时返回 None）
  - record_translation_feedback → LearningFeedback   写入本地 learning_feedback.json

【输出到哪里】
  → learning_feedback.json（本地）
  → 翻译官 Agent 可选调用 get_morph_correction 修正目标

【怎么调用】
  from agents.learning_agent import LearningAgent
  learning = LearningAgent()

  # 翻译官内部（可选）
  correction = learning.get_morph_correction(...)

  # 人工确认形态后（orchestrator 自动调用）
  learning.record_translation_feedback(...)
================================================================================
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import KB_DIR, MORPH_KEYS
from schemas.models import LearningFeedback


LEARNING_STORE = KB_DIR / "learning_feedback.json"


class LearningAgent:
    """学习 Agent 占位实现：记录反馈，暂不主动修正预测。"""

    def __init__(self, store_path: Path | None = None):
        self.store_path = store_path or LEARNING_STORE
        self._ensure_store()

    def _ensure_store(self) -> None:
        if not self.store_path.exists():
            self.store_path.write_text(
                json.dumps({"feedbacks": [], "enabled": False}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _load(self) -> dict:
        self._ensure_store()
        return json.loads(self.store_path.read_text(encoding="utf-8"))

    def _save(self, data: dict) -> None:
        self.store_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def is_enabled(self) -> bool:
        """是否启用学习修正（默认 False，仅占位记录）。"""
        return bool(self._load().get("enabled", False))

    def get_morph_correction(
        self,
        experience_baseline: dict,
        experience_targets: dict,
        baseline_metrics: dict,
        predicted_target: dict,
    ) -> dict[str, float] | None:
        """
        查询历史学习结果，返回对 predicted_target 的修正建议。
        占位实现：enabled=False 时始终返回 None。
        """
        if not self.is_enabled():
            return None

        # TODO: 组员可在此实现基于历史 feedbacks 的修正逻辑
        data = self._load()
        feedbacks = data.get("feedbacks", [])
        if not feedbacks:
            return None

        # 简单占位：取最近一条相似反馈的平均修正量
        corrections: dict[str, list[float]] = {k: [] for k in MORPH_KEYS}
        for fb in feedbacks[-20:]:
            pred = fb.get("predicted_target_metrics") or {}
            human = fb.get("human_corrected_metrics") or {}
            for k in MORPH_KEYS:
                if k in pred and k in human:
                    corrections[k].append(float(human[k]) - float(pred[k]))

        result = {}
        for k, vals in corrections.items():
            if vals:
                avg = sum(vals) / len(vals)
                if abs(avg) > 0.005:
                    result[k] = float(predicted_target.get(k, 0)) + avg

        return result or None

    def record_translation_feedback(
        self,
        experience_baseline: dict,
        experience_targets: dict,
        predicted_target_metrics: dict,
        human_corrected_metrics: dict,
        session_id: str = "",
        accurate: bool | None = None,
        notes: str = "",
    ) -> LearningFeedback:
        """记录一次翻译反馈，供后续学习。"""
        if accurate is None:
            accurate = self._auto_judge_accuracy(
                predicted_target_metrics, human_corrected_metrics
            )

        feedback = LearningFeedback(
            session_id=session_id or str(uuid.uuid4()),
            experience_baseline=experience_baseline,
            experience_targets=experience_targets,
            predicted_target_metrics=predicted_target_metrics,
            human_corrected_metrics=human_corrected_metrics,
            accurate=accurate,
            notes=notes,
        )

        data = self._load()
        entry = feedback.model_dump()
        entry["recorded_at"] = datetime.now().isoformat()
        data.setdefault("feedbacks", []).append(entry)
        self._save(data)
        return feedback

    @staticmethod
    def _auto_judge_accuracy(predicted: dict, corrected: dict) -> bool:
        """若人工未大幅修改形态目标，视为翻译准确。"""
        for k in MORPH_KEYS:
            p = float(predicted.get(k, 0))
            c = float(corrected.get(k, 0))
            if abs(c - p) > 0.03 and k != "color_richness":
                return False
            if k == "color_richness" and abs(c - p) > 0.8:
                return False
        return True

    def get_stats(self) -> dict:
        """返回学习统计（供前端/调试）。"""
        data = self._load()
        feedbacks = data.get("feedbacks", [])
        accurate_count = sum(1 for f in feedbacks if f.get("accurate"))
        return {
            "enabled": data.get("enabled", False),
            "total_feedbacks": len(feedbacks),
            "accurate_count": accurate_count,
            "accuracy_rate": round(accurate_count / len(feedbacks), 2) if feedbacks else None,
        }


def run_learning_record(**kwargs) -> LearningFeedback:
    return LearningAgent().record_translation_feedback(**kwargs)


if __name__ == "__main__":
    agent = LearningAgent()
    fb = agent.record_translation_feedback(
        experience_baseline={"comfort": 3, "restoration": 3, "safety": 3, "pleasure": 3, "stay": 3},
        experience_targets={"comfort": 4, "restoration": 5, "safety": 3, "pleasure": 4, "stay": 4},
        predicted_target_metrics={"green_view": 0.30},
        human_corrected_metrics={"green_view": 0.35},
        notes="demo",
    )
    print(fb.model_dump_json(indent=2, ensure_ascii=False))
    print(agent.get_stats())
