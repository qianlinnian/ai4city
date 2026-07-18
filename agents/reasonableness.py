"""Task 2 七项形态目标的内部合理性检查（只报警，不改写结果）。"""

from __future__ import annotations

from typing import Any

from config import MORPH_BOUNDS, MORPH_KEYS


_MAX_SINGLE_CHANGE = {
    "green_view": 0.20,
    "blue_view": 0.12,
    "sky_view": 0.20,
    "built_ratio": 0.20,
    "color_richness": 6.0,
    "edge_density": 0.15,
    "skyline_variance": 0.12,
}


def evaluate_task2_target(
    baseline: dict[str, float],
    target: dict[str, float],
    scene_understanding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    for key in MORPH_KEYS:
        value = float(target[key])
        lower, upper = MORPH_BOUNDS[key]
        if not lower <= value <= upper:
            warnings.append(f"{key}={value} 超出理论范围 {lower}~{upper}")
        delta = value - float(baseline[key])
        if abs(delta) > _MAX_SINGLE_CHANGE[key]:
            warnings.append(
                f"{key} 单次变化 {delta:+.4f} 较大，需专家复核可实施性"
            )

    green_delta = float(target["green_view"]) - float(baseline["green_view"])
    sky_delta = float(target["sky_view"]) - float(baseline["sky_view"])
    if green_delta > 0.12 and sky_delta > 0.12:
        warnings.append("绿视率与天空可视率同时大幅提升，可能存在视域占比冲突")

    scene = scene_understanding or {}
    scene_status = str(scene.get("status") or "")
    built_delta = float(target["built_ratio"]) - float(baseline["built_ratio"])
    skyline_delta = float(target["skyline_variance"]) - float(
        baseline["skyline_variance"]
    )
    if scene_status == "ok":
        if not scene.get("water") and (
            float(target["blue_view"]) - float(baseline["blue_view"])
        ) > 0.03:
            warnings.append("场景清单未确认真实水体，蓝视率异常提升不得转化为新增水体")
        if (scene.get("buildings") or scene.get("fixed_regions")) and built_delta < -0.10:
            warnings.append("人造物占比下降可能与固定建筑或保持区域冲突")
        if scene.get("buildings") and abs(skyline_delta) > 0.03:
            warnings.append("天际线变化可能触碰固定建筑轮廓，Task 3 不得改动建筑体量")
        if scene.get("roads") and abs(built_delta) > 0.12:
            warnings.append("人造物占比大幅变化需确认不改变道路拓扑与主要通行空间")
    elif scene_status in {"degraded", "disabled"}:
        warnings.append("场景理解已降级，固定建筑、道路、水体与实施性仅能由专家复核")

    return {
        "status": "warning" if warnings else "ok",
        "warnings": warnings,
        "checked_metrics": list(MORPH_KEYS),
        "scene_status": scene_status or "not_provided",
    }


__all__ = ["evaluate_task2_target"]
