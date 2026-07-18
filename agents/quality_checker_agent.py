"""Compare externally supplied post-edit metrics with confirmed targets.

Task 1 is an offline batch job.  This agent therefore never opens an image or
imports the Task 1 extractor; callers pass the seven post-edit values read from
the metrics table when those values become available.
"""

from __future__ import annotations

from collections.abc import Mapping

from config import MORPH_BOUNDS, MORPH_KEYS, MORPH_LABELS_ZH
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


def validate_morph_metrics(
    values: Mapping[str, float] | MorphMetrics,
    *,
    field_name: str,
) -> dict[str, float]:
    raw = values.as_dict() if isinstance(values, MorphMetrics) else dict(values)
    missing = [key for key in MORPH_KEYS if key not in raw]
    if missing:
        raise ValueError(f"{field_name}缺少指标：{', '.join(missing)}")

    result: dict[str, float] = {}
    for key in MORPH_KEYS:
        try:
            value = float(raw[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name}.{key} 必须是数值") from exc
        lower, upper = MORPH_BOUNDS[key]
        if not lower <= value <= upper:
            raise ValueError(
                f"{field_name}.{key}={value} 超出允许范围 [{lower}, {upper}]"
            )
        result[key] = value
    return result


class QualityCheckerAgent:
    """Build a quality report from metrics already calculated offline."""

    def run(
        self,
        measured_metrics: Mapping[str, float] | MorphMetrics,
        target_metrics: Mapping[str, float] | MorphMetrics,
        thresholds: Mapping[str, float] | None = None,
    ) -> QualityReport:
        measured = validate_morph_metrics(measured_metrics, field_name="修改后指标")
        target = validate_morph_metrics(target_metrics, field_name="目标指标")
        threshold_values = {**DEFAULT_THRESHOLDS, **dict(thresholds or {})}

        deviations: dict[str, float] = {}
        failures: list[str] = []
        for key in MORPH_KEYS:
            difference = measured[key] - target[key]
            deviations[key] = round(difference, 4)
            threshold = float(threshold_values.get(key, 0.08))
            if threshold < 0:
                raise ValueError(f"阈值 {key} 不能小于 0")
            if abs(difference) > threshold:
                failures.append(
                    f"{MORPH_LABELS_ZH.get(key, key)}偏差 {difference:+.3f}"
                    f"（阈值±{threshold:g}）"
                )

        passed = not failures
        details = "全部指标在允许偏差内。" if passed else "未达标：" + "；".join(failures)
        return QualityReport(
            measured_metrics=MorphMetrics(**measured),
            target_metrics=MorphMetrics(**target),
            deviations=deviations,
            passed=passed,
            details=details,
        )


def run_quality_check(
    measured_metrics: Mapping[str, float] | MorphMetrics,
    target_metrics: Mapping[str, float] | MorphMetrics,
    thresholds: Mapping[str, float] | None = None,
) -> QualityReport:
    return QualityCheckerAgent().run(measured_metrics, target_metrics, thresholds)


__all__ = [
    "DEFAULT_THRESHOLDS",
    "QualityCheckerAgent",
    "run_quality_check",
    "validate_morph_metrics",
]
