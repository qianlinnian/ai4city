"""
================================================================================
质检员 Agent（Quality Checker Agent）
文件: agents/quality_checker_agent.py
--------------------------------------------------------------------------------
【角色】
  对「修改后的全景图」重新提取形态要素，与制图员/人工确认的目标对比，
  输出达标报告；偏差可交给前端人工修正后写入知识库。

【输入】
  - edited_image_path: str     文生图结果路径
  - target_metrics: dict       目标形态要素
  - thresholds: dict | None    各指标允许绝对偏差（可选）

【输出】
  - QualityReport
      .measured_metrics
      .target_metrics
      .deviations
      .passed
      .details

【输出到哪里】
  → 前端展示修改后全景 + 指标对比
  → 若人工修正偏差指标，传给 memory_agent.store_feedback

【怎么调用】
  from agents.quality_checker_agent import QualityCheckerAgent
  report = QualityCheckerAgent().run(edited_image_path, target_metrics)
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import MORPH_KEYS, MORPH_LABELS_ZH
from morph_metrics_extractor import MorphMetricsExtractor
from schemas.models import MorphMetrics, QualityReport


DEFAULT_THRESHOLDS = {
    "green_view": 0.06,
    "blue_view": 0.04,
    "sky_view": 0.06,
    "built_ratio": 0.08,
    "edge_density": 0.04,
    "color_richness": 1.5,
    "skyline_variance": 0.02,
}


class QualityCheckerAgent:
    def __init__(self, extractor: MorphMetricsExtractor | None = None, force_fallback: bool = False):
        self.extractor = extractor or MorphMetricsExtractor(force_fallback=force_fallback)

    def run(
        self,
        edited_image_path: str | Path,
        target_metrics: dict | MorphMetrics,
        thresholds: dict | None = None,
    ) -> QualityReport:
        if isinstance(target_metrics, MorphMetrics):
            target = target_metrics.as_dict()
        else:
            target = {k: float(target_metrics[k]) for k in MORPH_KEYS if k in target_metrics}
            # 补齐缺失键
            for k in MORPH_KEYS:
                target.setdefault(k, 0.0)

        measured = self.extractor.calculate(edited_image_path)
        measured_d = measured.as_dict()
        thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

        deviations = {}
        fails = []
        for k in MORPH_KEYS:
            if k not in target:
                continue
            diff = measured_d[k] - float(target[k])
            deviations[k] = round(diff, 4)
            if abs(diff) > thr.get(k, 0.08):
                fails.append(
                    f"{MORPH_LABELS_ZH.get(k, k)} 偏差 {diff:+.3f}（阈值±{thr.get(k, 0.08)}）"
                )

        passed = len(fails) == 0
        details = "全部指标在允许偏差内。" if passed else ("未达标: " + "; ".join(fails))

        return QualityReport(
            measured_metrics=measured,
            target_metrics=MorphMetrics(**{k: float(target.get(k, 0)) for k in MORPH_KEYS}),
            deviations=deviations,
            passed=passed,
            details=details,
        )


def run_quality_check(edited_image_path: str, target_metrics: dict) -> QualityReport:
    return QualityCheckerAgent().run(edited_image_path, target_metrics)


if __name__ == "__main__":
    print("请通过 pipeline 或传入 edited_image_path 调用 QualityCheckerAgent.run")
