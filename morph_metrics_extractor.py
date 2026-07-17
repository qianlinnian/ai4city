"""
================================================================================
形态要素解析工具（独立可运行）
文件: morph_metrics_extractor.py
--------------------------------------------------------------------------------
【依据】code/数据指标提取技术路线(1).pdf + 指标定义及计算方式(1).xlsx

【做什么】
  输入一张 2:1 全景 JPG/PNG，计算 7 个可行形态要素指标：
  1. 绿视率 green_view
  2. 蓝视率 blue_view
  3. 天空可视率 sky_view
  4. 人造物占比 built_ratio
  5. 色彩丰富度 color_richness
  6. 边缘密度 edge_density
  7. 天际线变化率 skyline_variance

【怎么调用】
  1) 作为库：
       from morph_metrics_extractor import MorphMetricsExtractor
       extractor = MorphMetricsExtractor()
       result = extractor.calculate("path/to/pano.jpg")

  2) 命令行：
       python morph_metrics_extractor.py path/to/pano.jpg
       python morph_metrics_extractor.py path/to/pano.jpg --fallback   # 强制纯 OpenCV

【输出到哪里】
  - 返回 dict / MorphMetrics；命令行默认打印 JSON，可用 -o 写出文件

【技术路线】
  - 优先 SegFormer (ADE20K) 做语义分割占比类指标
  - 无 torch/transformers 或 --fallback 时，用 HSV 颜色启发式兜底（Demo 可用）
================================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

# 允许从 code/ 根目录导入
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import MORPH_LABELS_ZH, SEGFORMER_MODEL  # noqa: E402
from schemas.models import MorphMetrics  # noqa: E402


# ADE20K 常用类别（SegFormer-ADE20K）
ADE_SKY = 2
ADE_BUILDING = 1
ADE_TREE = 4
ADE_GRASS = 9
ADE_EARTH = 13
ADE_ROAD = 6
ADE_SIDEWALK = 11
ADE_PERSON = 12
ADE_CAR = 20
ADE_WATER = 21
ADE_FENCE = 32
ADE_WALL = 0  # wall 在部分映射中为 0 需谨慎；用 isin 列表覆盖
ADE_BUILT_IDS = [1, 6, 11, 20, 32, 43, 52, 61]  # building/road/sidewalk/car/fence/pillar/...


class MorphMetricsExtractor:
    """全景图形态要素一键提取器。"""

    def __init__(self, model_name: str = SEGFORMER_MODEL, force_fallback: bool = False):
        self.model_name = model_name
        self.force_fallback = force_fallback
        self._processor = None
        self._model = None
        self._seg_ready = False
        if not force_fallback:
            self._try_load_segformer()

    def _try_load_segformer(self) -> None:
        try:
            import torch
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

            self._torch = torch
            self._processor = SegformerImageProcessor.from_pretrained(self.model_name)
            self._model = SegformerForSemanticSegmentation.from_pretrained(self.model_name)
            self._model.eval()
            self._seg_ready = True
            print(f"[MorphMetrics] SegFormer 已加载: {self.model_name}")
        except Exception as e:
            print(f"[MorphMetrics] SegFormer 不可用，将使用 OpenCV 启发式: {e}")
            self._seg_ready = False

    @staticmethod
    def _imread_unicode(image_path: Path) -> Optional[np.ndarray]:
        """兼容中文路径的 BGR 读取。"""
        try:
            pil_img = Image.open(image_path).convert("RGB")
            rgb = np.array(pil_img)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            data = np.fromfile(str(image_path), dtype=np.uint8)
            return cv2.imdecode(data, cv2.IMREAD_COLOR)

    # ------------------------------------------------------------------ #
    # 公开接口
    # ------------------------------------------------------------------ #
    def calculate(self, image_path: str | Path) -> MorphMetrics:
        """
        输入: 全景图路径
        输出: MorphMetrics（比例为 0~1，color_richness 为有效色数）
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        # Windows 下 cv2.imread 无法可靠处理中文路径，统一经 PIL 读取
        img_bgr = self._imread_unicode(image_path)
        if img_bgr is None:
            raise ValueError(f"无法读取图像: {image_path}")

        h, w = img_bgr.shape[:2]
        total = float(h * w)

        if self._seg_ready and not self.force_fallback:
            sky_mask, green_mask, water_mask, built_mask = self._segment_masks(image_path, h, w)
        else:
            sky_mask, green_mask, water_mask, built_mask = self._heuristic_masks(img_bgr)

        sky_ratio = float(np.sum(sky_mask) / total)
        green_ratio = float(np.sum(green_mask) / total)
        blue_ratio = float(np.sum(water_mask) / total)
        built_ratio = float(np.sum(built_mask) / total)
        edge_density = self._edge_density(img_bgr)
        color_richness = self._color_richness(img_bgr)
        skyline_var = self._skyline_variance(sky_mask, h, w)

        return MorphMetrics(
            green_view=green_ratio,
            blue_view=blue_ratio,
            sky_view=sky_ratio,
            built_ratio=built_ratio,
            edge_density=edge_density,
            color_richness=color_richness,
            skyline_variance=skyline_var,
        )

    def calculate_dict(self, image_path: str | Path, percent: bool = False) -> dict[str, Any]:
        metrics = self.calculate(image_path)
        if percent:
            display = metrics.as_percent_display()
            return {
                MORPH_LABELS_ZH.get(k, k): display[k] for k in metrics.as_dict()
            }
        return metrics.as_dict()

    # ------------------------------------------------------------------ #
    # SegFormer 分割
    # ------------------------------------------------------------------ #
    def _segment_masks(self, image_path: Path, h: int, w: int):
        torch = self._torch
        pil_img = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=pil_img, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model(**inputs)
        logits = outputs.logits
        upsampled = torch.nn.functional.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False
        )
        seg = upsampled.argmax(dim=1)[0].cpu().numpy()

        sky_mask = seg == ADE_SKY
        green_mask = (seg == ADE_TREE) | (seg == ADE_GRASS) | (seg == ADE_EARTH)
        # plant / palm 等若存在：ADE20K plant=17
        green_mask = green_mask | (seg == 17)
        water_mask = (seg == ADE_WATER) | (seg == 26) | (seg == 60)  # water / sea / river-ish
        built_mask = np.isin(seg, ADE_BUILT_IDS) | (seg == 0)  # wall often 0 in ADE
        return sky_mask, green_mask, water_mask, built_mask

    # ------------------------------------------------------------------ #
    # OpenCV 启发式（无模型时 Demo 兜底）
    # ------------------------------------------------------------------ #
    def _heuristic_masks(self, img_bgr: np.ndarray):
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, w = img_bgr.shape[:2]

        # 天空：上半部高亮低饱和偏蓝
        upper = np.zeros((h, w), dtype=bool)
        upper[: h // 2, :] = True
        sky = (
            upper
            & (hsv[:, :, 2] > 140)
            & (hsv[:, :, 1] < 80)
            & ((hsv[:, :, 0] < 30) | (hsv[:, :, 0] > 90))
        )

        # 绿色植被
        green = (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 95) & (hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 40)

        # 水体偏蓝
        water = (hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 130) & (hsv[:, :, 1] > 50) & (hsv[:, :, 2] > 50)

        # 人造物粗估：非天空非绿非水
        built = ~(sky | green | water)
        return sky, green, water, built

    @staticmethod
    def _edge_density(img_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        return float(np.sum(edges > 0) / edges.size)

    @staticmethod
    def _color_richness(img_bgr: np.ndarray) -> float:
        """HSV 色相熵 → 有效色彩数量 N_effective = exp(H)"""
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        valid = (s_ch > 25) & (v_ch > 25)
        valid_h = h_ch[valid]
        if valid_h.size == 0:
            return 1.0
        counts, _ = np.histogram(valid_h, bins=24, range=(0, 180))
        # 仅保留占比 >= 0.5% 的色相参与熵（与指标表一致）
        probs = counts.astype(np.float64) / valid_h.size
        probs = probs[probs >= 0.005]
        if probs.size == 0:
            return 1.0
        probs = probs / probs.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        return float(np.exp(entropy))

    @staticmethod
    def _skyline_variance(sky_mask: np.ndarray, h: int, w: int) -> float:
        """SVR = sum(|y_{x+1}-y_x|) / ((W-1)*H)"""
        ys = []
        for col in range(w):
            idxs = np.where(sky_mask[:, col])[0]
            ys.append(int(idxs[-1]) if len(idxs) else h)
        ys = np.asarray(ys, dtype=np.float64)
        diffs = np.abs(np.diff(ys))
        return float(np.sum(diffs) / ((w - 1) * h)) if w > 1 else 0.0


def main():
    parser = argparse.ArgumentParser(description="全景图形态要素解析工具")
    parser.add_argument("image", help="全景 JPG/PNG 路径")
    parser.add_argument("--fallback", action="store_true", help="强制使用 OpenCV 启发式")
    parser.add_argument("-o", "--output", help="将 JSON 结果写入该文件")
    parser.add_argument("--percent", action="store_true", help="以百分比中文标签输出")
    args = parser.parse_args()

    extractor = MorphMetricsExtractor(force_fallback=args.fallback)
    result = extractor.calculate_dict(args.image, percent=args.percent)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"已写入: {args.output}")


if __name__ == "__main__":
    main()
