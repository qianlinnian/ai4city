"""全景街景七项形态指标批量提取。

本脚本保留现有 GluonCV Cityscapes 流程作为默认兼容模式，同时提供更高精度、
类别更完整的 ADE20K 模型配置。默认不允许自动下载模型，避免误触大文件下载。

指标口径以《指标定义及计算方式.xlsx》和项目已确认口径为准：
1. 比例类指标输出 0~1 的数值；
2. 色彩丰富度输出 0~24 的有效色相类别数，不再归一化到 0~1；
3. 边缘密度为边缘像素数 / 有效像素数；
4. 天际线变化率内部输出 0~1，显示时可格式化为百分比。
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_IMAGE_DIR = Path("./resized_images")
DEFAULT_OUTPUT_DIR = Path("./output_results")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}

CITYSCAPES_CLASSES = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)


@dataclass(frozen=True)
class ModelProfile:
    key: str
    backend: str
    model_name: str
    dataset: str
    description: str


MODEL_PROFILES: Mapping[str, ModelProfile] = {
    "cityscapes-legacy": ModelProfile(
        key="cityscapes-legacy",
        backend="gluoncv",
        model_name="deeplab_resnet101_citys",
        dataset="cityscapes",
        description="当前兼容模式；城市街景分割稳定，但没有水体类别。",
    ),
    "ade20k-resnest101": ModelProfile(
        key="ade20k-resnest101",
        backend="gluoncv",
        model_name="deeplab_resnest101_ade",
        dataset="ade20k",
        description="同一 MXNet 环境中的高精度方案；ADE20K 类别含天空和多种水体。",
    ),
    "segformer-b2-ade20k": ModelProfile(
        key="segformer-b2-ade20k",
        backend="segformer",
        model_name="nvidia/segformer-b2-finetuned-ade-512-512",
        dataset="ade20k",
        description="PyTorch/Transformers 平衡方案；需在 ai4city-mas 环境运行。",
    ),
    "segformer-b5-ade20k": ModelProfile(
        key="segformer-b5-ade20k",
        backend="segformer",
        model_name="nvidia/segformer-b5-finetuned-ade-640-640",
        dataset="ade20k",
        description="PyTorch/Transformers 高精度方案；CPU 推理较慢。",
    ),
}


# Cityscapes 映射保持现有脚本口径，避免越权改动学长正在核对的绿/蓝视率逻辑。
CITYSCAPES_GROUPS = {
    "green": {8, 9},
    "blue": set(),
    "sky": {10},
    "built": {0, 1, 2, 3, 4, 5, 6, 7, 13, 14, 15, 16, 17, 18},
}

# ADE20K 的类号来自官方 150 类标签。此映射用于可选高精度配置；在学长完成
# 绿/蓝视率最终核对前，不将它替换为默认配置。
ADE20K_GROUPS = {
    "green": {4, 9, 17, 29, 66, 72, 125},
    "blue": {21, 26, 60, 104, 109, 113, 128},
    "sky": {2},
    "built": {
        0, 1, 3, 6, 8, 11, 14, 20, 25, 32, 36, 38, 40, 42, 43, 48,
        51, 52, 53, 54, 59, 61, 69, 76, 80, 83, 84, 86, 87, 88, 90,
        93, 95, 100, 102, 103, 105, 106, 114, 116, 121, 122, 123, 127,
        132, 136, 138, 140, 144,
    },
}


@dataclass(frozen=True)
class OutputDirs:
    root: Path
    segmentation: Path
    edges: Path
    skyline: Path


@dataclass(frozen=True)
class SkylineMetrics:
    variation_rate: float
    valid_column_ratio: float
    raw_y: np.ndarray
    smoothed_y: np.ndarray | None


def prepare_output_dirs(root: Path) -> OutputDirs:
    dirs = OutputDirs(
        root=root,
        segmentation=root / "segmentation_results",
        edges=root / "edge_density_maps",
        skyline=root / "skyline_boundary_maps",
    )
    for path in (dirs.root, dirs.segmentation, dirs.edges, dirs.skyline):
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _choose_odd_window(value: int, upper_bound: int) -> int:
    if upper_bound < 3:
        return 1
    value = max(3, min(int(value), upper_bound))
    if value % 2 == 0:
        value = value - 1 if value == upper_bound else value + 1
    return max(3, value)


def _read_bgr(image_or_path: np.ndarray | str | Path) -> np.ndarray:
    if isinstance(image_or_path, np.ndarray):
        image = image_or_path
    else:
        image = cv2.imread(str(image_or_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"无法读取图像: {image_or_path}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("输入必须是三通道 BGR 图像")
    return image


def _pascal_palette(size: int = 256) -> np.ndarray:
    """生成确定性的类别颜色表，避免 JPEG 调色板模式写入错误。"""
    palette = np.zeros((size, 3), dtype=np.uint8)
    for label in range(size):
        value = label
        for shift in range(8):
            palette[label, 0] |= ((value >> 0) & 1) << (7 - shift)
            palette[label, 1] |= ((value >> 1) & 1) << (7 - shift)
            palette[label, 2] |= ((value >> 2) & 1) << (7 - shift)
            value >>= 3
    return palette


def save_segmentation_map(prediction: np.ndarray, output_path: Path) -> None:
    palette = _pascal_palette()
    rgb = palette[prediction.astype(np.int64) % len(palette)]
    if not cv2.imwrite(str(output_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"语义分割图写入失败: {output_path}")


class GluonCVSegmenter:
    def __init__(
        self,
        profile: ModelProfile,
        device: str,
        allow_download: bool,
    ) -> None:
        # MXNet 1.x 仍会访问 np.bool；只在加载 MXNet 后端时做兼容补丁。
        np.__dict__.setdefault("bool", np.bool_)

        import mxnet as mx
        import gluoncv
        from gluoncv.data.transforms.presets.segmentation import test_transform
        from gluoncv.model_zoo.model_store import _model_sha1

        self.mx = mx
        self.test_transform = test_transform
        self.profile = profile

        if device == "gpu" and mx.context.num_gpus() == 0:
            raise RuntimeError("指定了 GPU，但当前 MXNet 环境未检测到 GPU")
        use_gpu = device == "gpu" or (device == "auto" and mx.context.num_gpus() > 0)
        self.ctx = mx.gpu(0) if use_gpu else mx.cpu(0)

        sha1 = _model_sha1.get(profile.model_name)
        cache_path = (
            Path.home() / ".mxnet" / "models" / f"{profile.model_name}-{sha1[:8]}.params"
            if sha1
            else None
        )
        if not allow_download and (cache_path is None or not cache_path.exists()):
            raise FileNotFoundError(
                f"模型未缓存: {profile.model_name}。如确认要下载，请增加 "
                "--allow-model-download；本脚本默认禁止静默下载。"
            )

        print(f"正在载入模型 {profile.model_name}，设备 {self.ctx} ...")
        self.model = gluoncv.model_zoo.get_model(
            profile.model_name,
            pretrained=True,
            ctx=self.ctx,
        )

        if profile.dataset == "cityscapes":
            self.class_names = CITYSCAPES_CLASSES
        else:
            from gluoncv.data.ade20k.segmentation import ADE20KSegmentation

            self.class_names = tuple(ADE20KSegmentation.CLASSES)

    def predict(self, image_path: Path) -> np.ndarray:
        img = self.mx.image.imread(str(image_path))
        original_h, original_w = int(img.shape[0]), int(img.shape[1])
        transformed = self.test_transform(img, ctx=self.ctx)
        output = self.model.predict(transformed)
        prediction = (
            self.mx.nd.squeeze(self.mx.nd.argmax(output, 1))
            .asnumpy()
            .astype(np.int32)
        )
        if prediction.shape != (original_h, original_w):
            prediction = cv2.resize(
                prediction,
                (original_w, original_h),
                interpolation=cv2.INTER_NEAREST,
            )
        return prediction


class SegFormerSegmenter:
    def __init__(
        self,
        profile: ModelProfile,
        device: str,
        allow_download: bool,
    ) -> None:
        import torch
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

        self.torch = torch
        self.profile = profile
        if device == "gpu" and not torch.cuda.is_available():
            raise RuntimeError("指定了 GPU，但当前 PyTorch 环境未检测到 CUDA")
        use_gpu = device == "gpu" or (device == "auto" and torch.cuda.is_available())
        self.device = torch.device("cuda:0" if use_gpu else "cpu")

        local_only = not allow_download
        print(f"正在载入模型 {profile.model_name}，设备 {self.device} ...")
        try:
            self.processor = AutoImageProcessor.from_pretrained(
                profile.model_name,
                local_files_only=local_only,
            )
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                profile.model_name,
                local_files_only=local_only,
            ).to(self.device)
        except OSError as exc:
            if local_only:
                raise FileNotFoundError(
                    f"模型未缓存: {profile.model_name}。如确认要下载，请增加 "
                    "--allow-model-download；本脚本默认禁止静默下载。"
                ) from exc
            raise
        self.model.eval()
        id2label = self.model.config.id2label
        self.class_names = tuple(
            str(id2label.get(index, id2label.get(str(index), f"class_{index}")))
            for index in range(self.model.config.num_labels)
        )

    def predict(self, image_path: Path) -> np.ndarray:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        target_h, target_w = image.height, image.width
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits
            logits = self.torch.nn.functional.interpolate(
                logits,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
        return logits.argmax(dim=1)[0].cpu().numpy().astype(np.int32)


def build_segmenter(
    profile: ModelProfile,
    device: str,
    allow_download: bool,
) -> GluonCVSegmenter | SegFormerSegmenter:
    if profile.backend == "gluoncv":
        return GluonCVSegmenter(profile, device, allow_download)
    if profile.backend == "segformer":
        return SegFormerSegmenter(profile, device, allow_download)
    raise ValueError(f"不支持的模型后端: {profile.backend}")


def calculate_color_richness(
    image_or_path: np.ndarray | str | Path,
    hue_bins: int = 24,
    saturation_threshold: float = 0.1,
    min_area_ratio: float = 0.005,
) -> int:
    """计算有效色相类别数 N_color，范围 0~hue_bins。"""
    if hue_bins <= 0:
        raise ValueError("hue_bins 必须大于 0")
    if not 0 <= saturation_threshold <= 1:
        raise ValueError("saturation_threshold 必须在 0~1")
    if not 0 <= min_area_ratio <= 1:
        raise ValueError("min_area_ratio 必须在 0~1")

    image = _read_bgr(image_or_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, _ = cv2.split(hsv)
    threshold_8bit = int(math.ceil(saturation_threshold * 255))
    valid_mask = saturation >= threshold_8bit
    valid_count = int(np.count_nonzero(valid_mask))
    if valid_count == 0:
        return 0

    hist, _ = np.histogram(
        hue[valid_mask],
        bins=np.linspace(0, 180, hue_bins + 1),
    )
    proportions = hist.astype(np.float64) / valid_count
    return int(np.count_nonzero(proportions >= min_area_ratio))


def _filter_small_edge_components(edges: np.ndarray, min_pixels: int) -> np.ndarray:
    if min_pixels <= 1:
        return edges
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (edges > 0).astype(np.uint8),
        connectivity=8,
    )
    keep = np.zeros(count, dtype=bool)
    if count > 1:
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_pixels
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def calculate_edge_density_and_save(
    image_or_path: np.ndarray | str | Path,
    output_path: Path | None = None,
    analysis_width: int = 1024,
    canny_low: int = 100,
    canny_high: int = 200,
    min_component_pixels: int = 20,
) -> float:
    """在统一尺度上提取主边缘，返回边缘像素 / 有效像素。"""
    if analysis_width <= 0:
        raise ValueError("analysis_width 必须大于 0")
    if not 0 <= canny_low < canny_high <= 255:
        raise ValueError("Canny 阈值必须满足 0 <= low < high <= 255")

    image = _read_bgr(image_or_path)
    height, width = image.shape[:2]
    if width > analysis_width:
        scale = analysis_width / width
        image = cv2.resize(
            image,
            (analysis_width, max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), sigmaX=1.4, sigmaY=1.4)
    edges = cv2.Canny(blurred, canny_low, canny_high, L2gradient=True)
    edges = _filter_small_edge_components(edges, min_component_pixels)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), edges):
            raise OSError(f"边缘图写入失败: {output_path}")

    return float(np.count_nonzero(edges) / edges.size)


def _clean_top_sky_mask(
    prediction: np.ndarray,
    sky_ids: Iterable[int],
    min_component_ratio: float = 0.0005,
) -> np.ndarray:
    """保留靠近图像顶部的天空连通域，并修补小孔洞与孤立噪声。"""
    sky_ids = tuple(int(value) for value in sky_ids)
    if not sky_ids:
        return np.zeros(prediction.shape, dtype=bool)

    sky = np.isin(prediction, sky_ids).astype(np.uint8) * 255
    height, width = sky.shape
    kernel_size = _choose_odd_window(round(min(height, width) * 0.007), min(height, width))
    if kernel_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        sky = cv2.morphologyEx(sky, cv2.MORPH_CLOSE, kernel)
        sky = cv2.morphologyEx(sky, cv2.MORPH_OPEN, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (sky > 0).astype(np.uint8),
        connectivity=8,
    )
    if count <= 1:
        return np.zeros_like(sky, dtype=bool)

    min_area = max(16, round(height * width * min_component_ratio))
    top_band = max(1, round(height * 0.03))
    detached_min_area = max(min_area, round(height * width * 0.002))
    keep_labels = [
        label
        for label in range(1, count)
        if (
            stats[label, cv2.CC_STAT_AREA] >= min_area
            and stats[label, cv2.CC_STAT_TOP] <= top_band
        )
        or (
            # 树枝、电线可能把主天空区域与图像顶边切断。只要连通域足够大、
            # 且起点位于图像上方四分之一，仍视为可信天空候选。
            stats[label, cv2.CC_STAT_AREA] >= detached_min_area
            and stats[label, cv2.CC_STAT_TOP] <= round(height * 0.25)
        )
    ]

    # 顶部被树冠等短暂遮挡时，允许使用面积足够大且起点位于上方 15% 的天空。
    if not keep_labels:
        candidate = int(1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if (
            stats[candidate, cv2.CC_STAT_AREA] >= max(min_area, round(height * width * 0.01))
            and stats[candidate, cv2.CC_STAT_TOP] <= round(height * 0.15)
        ):
            keep_labels = [candidate]

    return np.isin(labels, keep_labels)


def _circular_interpolate(values: np.ndarray) -> np.ndarray | None:
    valid = np.flatnonzero(np.isfinite(values))
    if valid.size < 2:
        return None
    width = values.size
    x = np.concatenate(([valid[-1] - width], valid, [valid[0] + width]))
    y = np.concatenate(([values[valid[-1]]], values[valid], [values[valid[0]]]))
    return np.interp(np.arange(width), x, y)


def _circular_median(values: np.ndarray, window: int) -> np.ndarray:
    window = _choose_odd_window(window, values.size)
    if window <= 1:
        return values.astype(np.float64, copy=True)
    radius = window // 2
    # ai4city-mxnet 使用的旧版 NumPy 没有 sliding_window_view；rolling
    # 堆叠在 2K~8K 宽全景上内存规模可控，同时保持首尾循环窗口。
    windows = np.stack(
        [np.roll(values, shift) for shift in range(-radius, radius + 1)],
        axis=0,
    )
    return np.median(windows, axis=0)


def _circular_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = _choose_odd_window(window, values.size)
    if window <= 1:
        return values.astype(np.float64, copy=True)
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="wrap")
    kernel = np.full(window, 1.0 / window, dtype=np.float64)
    return np.convolve(padded, kernel, mode="valid")


def calculate_skyline_and_save(
    prediction: np.ndarray,
    image_or_path: np.ndarray | str | Path,
    sky_ids: Iterable[int],
    output_path: Path | None = None,
    smoothing_ratio: float = 0.015,
) -> SkylineMetrics:
    """提取去噪天际线并按表格的一阶差分公式计算 0~1 变化率。"""
    if prediction.ndim != 2:
        raise ValueError("prediction 必须是二维类别矩阵")
    if not 0 < smoothing_ratio <= 0.2:
        raise ValueError("smoothing_ratio 必须在 (0, 0.2]")

    image = _read_bgr(image_or_path)
    if prediction.shape != image.shape[:2]:
        prediction = cv2.resize(
            prediction.astype(np.int32),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    cleaned_sky = _clean_top_sky_mask(prediction, sky_ids)
    height, width = cleaned_sky.shape
    raw_y = np.full(width, np.nan, dtype=np.float64)
    for x in range(width):
        sky_pixels = np.flatnonzero(cleaned_sky[:, x])
        if sky_pixels.size:
            raw_y[x] = float(sky_pixels[-1])

    valid_ratio = float(np.count_nonzero(np.isfinite(raw_y)) / width)
    interpolated = _circular_interpolate(raw_y)
    if interpolated is None or width < 2:
        result = SkylineMetrics(0.0, valid_ratio, raw_y, None)
    else:
        median_window = _choose_odd_window(round(width * smoothing_ratio), width)
        smoothed = _circular_median(interpolated, median_window)
        average_window = _choose_odd_window(max(3, median_window // 3), width)
        smoothed = _circular_moving_average(smoothed, average_window)
        diff_sum = float(np.abs(np.diff(smoothed)).sum())
        variation_rate = diff_sum / ((width - 1) * height)
        result = SkylineMetrics(
            variation_rate=float(variation_rate),
            valid_column_ratio=valid_ratio,
            raw_y=raw_y,
            smoothed_y=smoothed,
        )

    if output_path is not None:
        overlay = image.copy()
        if result.smoothed_y is not None:
            all_points = np.column_stack(
                (np.arange(width), np.rint(result.smoothed_y).astype(np.int32))
            ).reshape((-1, 1, 2))
            # 橙色细线表示根据邻近列插值得到的完整曲线；只有原始天空有效列
            # 对应的连续区段绘制为红色粗线，避免把模型缺失伪装成可靠识别。
            cv2.polylines(overlay, [all_points], False, (0, 165, 255), 1, cv2.LINE_AA)
            valid_indices = np.flatnonzero(np.isfinite(result.raw_y))
            if valid_indices.size:
                split_at = np.flatnonzero(np.diff(valid_indices) > 1) + 1
                for run in np.split(valid_indices, split_at):
                    if run.size < 2:
                        continue
                    run_points = np.column_stack(
                        (run, np.rint(result.smoothed_y[run]).astype(np.int32))
                    ).reshape((-1, 1, 2))
                    cv2.polylines(overlay, [run_points], False, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(
            overlay,
            f"SVR={result.variation_rate:.4%}  coverage={result.valid_column_ratio:.1%}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), overlay):
            raise OSError(f"天际线图写入失败: {output_path}")

    return result


def _groups_for_dataset(dataset: str) -> Mapping[str, set[int]]:
    if dataset == "cityscapes":
        return CITYSCAPES_GROUPS
    if dataset == "ade20k":
        return ADE20K_GROUPS
    raise ValueError(f"没有为数据集 {dataset} 定义指标映射")


def _ratio_for_ids(prediction: np.ndarray, class_ids: Iterable[int]) -> float:
    class_ids = tuple(class_ids)
    if not class_ids:
        return 0.0
    return float(np.count_nonzero(np.isin(prediction, class_ids)) / prediction.size)


def export_metric_records(records: Sequence[Mapping[str, object]], output_root: Path) -> Path:
    """导出指标表；缺少 Excel 引擎时自动回退到 UTF-8 CSV。"""
    dataframe = pd.DataFrame(records)
    excel_path = output_root / "metrics_results.xlsx"
    try:
        dataframe.to_excel(excel_path, index=False)
        return excel_path
    except ModuleNotFoundError as exc:
        if exc.name not in {"openpyxl", "xlsxwriter"}:
            raise

    csv_path = output_root / "metrics_results.csv"
    dataframe.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(
        "当前 Python 环境没有 Excel 写入引擎，已自动改存 UTF-8 CSV；"
        "可直接用 Excel 打开。"
    )
    return csv_path


def process_pipeline(args: argparse.Namespace) -> Path | None:
    image_dir = Path(args.image_dir)
    output_dirs = prepare_output_dirs(Path(args.output_dir))
    if not image_dir.exists():
        print(f"错误：图像文件夹不存在: {image_dir}")
        return None

    image_paths = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        print(f"没有在 {image_dir} 中找到有效图片。")
        return None

    profile = MODEL_PROFILES[args.model_profile]
    print(f"模型配置: {profile.key} - {profile.description}")
    segmenter = build_segmenter(profile, args.device, args.allow_model_download)
    groups = _groups_for_dataset(profile.dataset)
    records: list[dict[str, object]] = []

    print(f"开始处理全景街景图片，共 {len(image_paths)} 张。")
    for image_path in tqdm(image_paths, desc="街景图像多指标提取中"):
        try:
            image = _read_bgr(image_path)
            prediction = segmenter.predict(image_path)
            stem = image_path.stem
            save_segmentation_map(
                prediction,
                output_dirs.segmentation / f"seg_{stem}.png",
            )

            raw_seg_results = {
                class_name: float(np.count_nonzero(prediction == class_id) / prediction.size)
                for class_id, class_name in enumerate(segmenter.class_names)
            }
            color_richness = calculate_color_richness(image)
            edge_density = calculate_edge_density_and_save(
                image,
                output_dirs.edges / f"edge_{stem}.png",
                analysis_width=args.edge_analysis_width,
                canny_low=args.canny_low,
                canny_high=args.canny_high,
                min_component_pixels=args.edge_min_component_pixels,
            )
            skyline = calculate_skyline_and_save(
                prediction,
                image,
                groups["sky"],
                output_dirs.skyline / f"skyline_{stem}.png",
                smoothing_ratio=args.skyline_smoothing_ratio,
            )

            record: dict[str, object] = {
                "图像名称": image_path.name,
                "模型配置": profile.key,
                "模型名称": profile.model_name,
                "综合-绿视率(GVI)": _ratio_for_ids(prediction, groups["green"]),
                "综合-蓝视率(BVI)": _ratio_for_ids(prediction, groups["blue"]),
                "综合-天空可视率": _ratio_for_ids(prediction, groups["sky"]),
                "综合-人造物占比": _ratio_for_ids(prediction, groups["built"]),
                "综合-色彩丰富度(CR)": color_richness,
                "综合-边缘密度(ED)": edge_density,
                "综合-天际线变化率(SVR)": skyline.variation_rate,
                "诊断-天际线有效列占比": skyline.valid_column_ratio,
            }
            for class_name, ratio in raw_seg_results.items():
                record[f"原始-{class_name}"] = ratio
            records.append(record)
        except Exception as exc:  # 单张失败不应中断其余批处理
            print(f"\n图片 {image_path.name} 处理出错，已跳过。错误原因: {exc}")

    if not records:
        print("没有成功处理的图片，不生成结果表。")
        return None

    result_path = export_metric_records(records, output_dirs.root)
    print(f"处理完成，指标结果: {result_path}")
    print(f"分割图、边缘图和天际线图: {output_dirs.root}")
    return result_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--model-profile",
        choices=tuple(MODEL_PROFILES),
        default="cityscapes-legacy",
        help="模型配置；默认使用已存在的 Cityscapes 兼容模型。",
    )
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="允许模型库下载未缓存权重；默认禁止。",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 张，用于验收。")
    parser.add_argument("--edge-analysis-width", type=int, default=1024)
    parser.add_argument("--canny-low", type=int, default=100)
    parser.add_argument("--canny-high", type=int, default=200)
    parser.add_argument("--edge-min-component-pixels", type=int, default=20)
    parser.add_argument("--skyline-smoothing-ratio", type=float, default=0.015)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit 必须大于 0")
    try:
        process_pipeline(args)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"错误：{exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
