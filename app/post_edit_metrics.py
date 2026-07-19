"""生成后图像的七项形态指标适配层。

本模块只编排既有 ``morph_metrics_extractor.py``：不改动分割、色彩、边缘或
天际线算法。所有生成的分析图写入项目 outputs，绝不写回原始数据目录。
"""

from __future__ import annotations

import hashlib
import importlib
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from config import (
    POST_EDIT_METRICS_ALLOW_MODEL_DOWNLOAD,
    POST_EDIT_METRICS_DEVICE,
    POST_EDIT_METRICS_MODEL_PROFILE,
    POST_EDIT_METRICS_OUTPUT_DIR,
)


_SEGMENTER_LOCK = threading.Lock()


@lru_cache(maxsize=4)
def _load_segmenter(
    profile_key: str,
    device: str,
    allow_model_download: bool,
) -> tuple[Any, Any, Any]:
    """懒加载并复用现有语义分割器，避免每次生成图都重新载入模型。"""
    extractor = importlib.import_module("morph_metrics_extractor")
    try:
        profile = extractor.MODEL_PROFILES[profile_key]
    except KeyError as exc:
        available = ", ".join(sorted(extractor.MODEL_PROFILES))
        raise ValueError(
            f"未知 POST_EDIT_METRICS_MODEL_PROFILE={profile_key!r}；可选：{available}"
        ) from exc
    try:
        segmenter = extractor.build_segmenter(profile, device, allow_model_download)
    except Exception as exc:
        raise RuntimeError(
            "生成后七项形态指标提取器不可用："
            f"{profile_key} 未能加载。"
            "默认禁止自动下载模型；如已确认允许下载，可设置 "
            "POST_EDIT_METRICS_ALLOW_MODEL_DOWNLOAD=true。原始错误："
            f"{exc}"
        ) from exc
    return extractor, profile, segmenter


def _artifact_root(image_path: Path) -> Path:
    stat = image_path.stat()
    fingerprint = hashlib.sha256(
        f"{image_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
    ).hexdigest()[:12]
    return POST_EDIT_METRICS_OUTPUT_DIR / f"{image_path.stem}_{fingerprint}"


def extract_post_edit_metrics(image_path: str | Path) -> dict[str, float]:
    """计算一张生成图的既有七项形态指标并保存派生分析图。

    返回值只包含主流程使用的七项内部 key；比例指标为 0～1，色彩丰富度为
    0～24。任何异常均向上抛出，供编排器记录为可见的提取失败，而不伪造数值。
    """
    source = Path(image_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"找不到待提取指标的生成图：{source}")

    extractor, profile, segmenter = _load_segmenter(
        POST_EDIT_METRICS_MODEL_PROFILE,
        POST_EDIT_METRICS_DEVICE,
        POST_EDIT_METRICS_ALLOW_MODEL_DOWNLOAD,
    )
    output_dirs = extractor.prepare_output_dirs(_artifact_root(source))
    image = extractor._read_bgr(source)
    # SegFormer 模型实例跨 Gradio 请求复用；推理阶段串行化以避免 GPU/CPU 会话竞争。
    with _SEGMENTER_LOCK:
        prediction = segmenter.predict(source)

    extractor.save_segmentation_map(
        prediction,
        output_dirs.segmentation / f"seg_{source.stem}.png",
    )
    groups = extractor._groups_for_dataset(profile.dataset)
    edge_density = extractor.calculate_edge_density_and_save(
        image,
        output_dirs.edges / f"edge_{source.stem}.png",
    )
    skyline = extractor.calculate_skyline_and_save(
        prediction,
        image,
        groups["sky"],
        output_dirs.skyline / f"skyline_{source.stem}.png",
    )
    return {
        "green_view": float(extractor._ratio_for_ids(prediction, groups["green"])),
        "blue_view": float(extractor._ratio_for_ids(prediction, groups["blue"])),
        "sky_view": float(extractor._ratio_for_ids(prediction, groups["sky"])),
        "built_ratio": float(extractor._ratio_for_ids(prediction, groups["built"])),
        "color_richness": float(extractor.calculate_color_richness(image)),
        "edge_density": float(edge_density),
        "skyline_variance": float(skyline.variation_rate),
    }


__all__ = ["extract_post_edit_metrics"]
