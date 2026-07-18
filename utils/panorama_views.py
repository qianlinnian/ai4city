"""等距柱状全景的概览图与普通透视图生成。

本模块只为 Task 2/3 的多图场景理解服务，不参与 Task 1 指标计算。
原始图片始终只读；派生图与元数据写入项目输出缓存。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
import cv2
import numpy as np

from config import (
    DATA_DIR,
    PANORAMA_ASPECT_TOLERANCE,
    PANORAMA_DOWNWARD_PITCH,
    PANORAMA_HORIZONTAL_YAWS,
    PANORAMA_INCLUDE_DOWNWARD,
    PANORAMA_OVERVIEW_HEIGHT,
    PANORAMA_OVERVIEW_WIDTH,
    PANORAMA_PERSPECTIVE_FOV,
    PANORAMA_PERSPECTIVE_HEIGHT,
    PANORAMA_PERSPECTIVE_WIDTH,
    PANORAMA_STRICT_ASPECT,
    PANORAMA_VIEW_CACHE_DIR,
)
from schemas.models import PanoramaViewMetadata, PanoramaViewSet


@dataclass(frozen=True)
class PanoramaProjectionConfig:
    overview_width: int = PANORAMA_OVERVIEW_WIDTH
    overview_height: int = PANORAMA_OVERVIEW_HEIGHT
    perspective_width: int = PANORAMA_PERSPECTIVE_WIDTH
    perspective_height: int = PANORAMA_PERSPECTIVE_HEIGHT
    fov: float = PANORAMA_PERSPECTIVE_FOV
    horizontal_yaws: tuple[float, ...] = PANORAMA_HORIZONTAL_YAWS
    include_downward: bool = PANORAMA_INCLUDE_DOWNWARD
    downward_pitch: float = PANORAMA_DOWNWARD_PITCH
    aspect_tolerance: float = PANORAMA_ASPECT_TOLERANCE
    strict_aspect: bool = PANORAMA_STRICT_ASPECT
    jpeg_quality: int = 90

    def validate(self) -> None:
        dimensions = (
            self.overview_width,
            self.overview_height,
            self.perspective_width,
            self.perspective_height,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("全景派生图宽高必须为正整数")
        if not 1.0 < self.fov < 179.0:
            raise ValueError("透视图 FOV 必须位于 1 到 179 度之间")
        if not self.horizontal_yaws:
            raise ValueError("至少需要一个水平 yaw")
        if self.aspect_tolerance < 0:
            raise ValueError("全景宽高比容差不能为负")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("JPEG 质量必须位于 1 到 100 之间")


def _read_image(path: Path) -> np.ndarray:
    """兼容 Windows 中文路径的只读图像加载。"""
    try:
        payload = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"无法读取全景图: {path}") from exc
    image = cv2.imdecode(payload, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法解码全景图: {path}")
    return image


def _write_jpeg(path: Path, image: np.ndarray, quality: int) -> None:
    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise OSError(f"无法编码派生图: {path}")
    path.write_bytes(encoded.tobytes())


def equirectangular_to_perspective(
    panorama: np.ndarray,
    *,
    yaw: float,
    pitch: float,
    fov: float,
    width: int,
    height: int,
) -> np.ndarray:
    """以球面射线投影生成普通透视图，并在左右接缝处环绕采样。"""
    if panorama.ndim not in {2, 3}:
        raise ValueError("全景图数组必须为 H×W 或 H×W×C")
    if width <= 0 or height <= 0:
        raise ValueError("输出宽高必须为正整数")
    if not 1.0 < float(fov) < 179.0:
        raise ValueError("FOV 必须位于 1 到 179 度之间")

    source_height, source_width = panorama.shape[:2]
    horizontal_tan = math.tan(math.radians(float(fov)) / 2.0)
    vertical_tan = horizontal_tan * float(height) / float(width)

    xs = np.linspace(-horizontal_tan, horizontal_tan, width, dtype=np.float32)
    ys = np.linspace(vertical_tan, -vertical_tan, height, dtype=np.float32)
    plane_x, plane_y = np.meshgrid(xs, ys)

    yaw_r = math.radians(float(yaw))
    pitch_r = math.radians(float(pitch))
    forward = np.array(
        [
            math.sin(yaw_r) * math.cos(pitch_r),
            math.sin(pitch_r),
            math.cos(yaw_r) * math.cos(pitch_r),
        ],
        dtype=np.float32,
    )
    right = np.array(
        [math.cos(yaw_r), 0.0, -math.sin(yaw_r)],
        dtype=np.float32,
    )
    up = np.cross(forward, right).astype(np.float32)

    rays = (
        forward[None, None, :]
        + plane_x[:, :, None] * right[None, None, :]
        + plane_y[:, :, None] * up[None, None, :]
    )
    rays /= np.linalg.norm(rays, axis=2, keepdims=True).clip(min=1e-8)

    longitude = np.arctan2(rays[:, :, 0], rays[:, :, 2])
    latitude = np.arcsin(np.clip(rays[:, :, 1], -1.0, 1.0))
    map_x = ((longitude / (2.0 * math.pi) + 0.5) * source_width).astype(
        np.float32
    )
    map_x = np.mod(map_x, float(source_width))
    map_y = ((0.5 - latitude / math.pi) * source_height).astype(np.float32)
    map_y = np.clip(map_y, 0.0, float(source_height - 1))

    return cv2.remap(
        panorama,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )


class PanoramaViewGenerator:
    """生成、缓存并记录一组可交给多模态模型的全景视图。"""

    def __init__(
        self,
        output_dir: str | Path = PANORAMA_VIEW_CACHE_DIR,
        config: PanoramaProjectionConfig | None = None,
        *,
        read_only_data_dir: str | Path = DATA_DIR,
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.read_only_data_dir = Path(read_only_data_dir).resolve()
        self.config = config or PanoramaProjectionConfig()
        self.config.validate()
        if self.output_dir == self.read_only_data_dir or self.output_dir.is_relative_to(
            self.read_only_data_dir
        ):
            raise ValueError("全景派生图缓存目录不得位于只读数据目录中")

    @staticmethod
    def _source_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _config_signature(self) -> str:
        payload = json.dumps(asdict(self.config), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]

    @staticmethod
    def _safe_stem(path: Path) -> str:
        stem = re.sub(r"[^0-9A-Za-z_.-]+", "_", path.stem).strip("._")
        return (stem or "panorama")[:80]

    @staticmethod
    def _yaw_label(yaw: float) -> str:
        normalized = float(yaw) % 360.0
        if abs(normalized - round(normalized)) < 1e-8:
            return f"{int(round(normalized)):03d}"
        return f"{normalized:06.2f}".replace(".", "p")

    def _metadata_if_cached(
        self,
        metadata_path: Path,
        source_sha256: str,
    ) -> PanoramaViewSet | None:
        if not metadata_path.is_file():
            return None
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            if data.get("source_sha256") != source_sha256:
                return None
            if data.get("config_signature") != self._config_signature():
                return None
            data.pop("config_signature", None)
            result = PanoramaViewSet(**data)
            if not result.views or not all(
                Path(view.output_path).is_file() for view in result.views
            ):
                return None
            return result.model_copy(update={"cache_hit": True})
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def generate(
        self,
        source_path: str | Path,
        *,
        source_image_id: str = "",
    ) -> PanoramaViewSet:
        source = Path(source_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"找不到原始全景图: {source}")

        source_sha256 = self._source_sha256(source)
        image_id = source_image_id or source.stem
        cache_key = f"{self._safe_stem(source)}_{source_sha256[:12]}_{self._config_signature()}"
        target_dir = self.output_dir / cache_key
        metadata_path = target_dir / "views.json"
        cached = self._metadata_if_cached(metadata_path, source_sha256)
        if cached is not None:
            return cached

        panorama = _read_image(source)
        source_height, source_width = panorama.shape[:2]
        ratio = source_width / source_height
        warning_messages: list[str] = []
        if abs(ratio - 2.0) > self.config.aspect_tolerance:
            message = (
                f"输入全景宽高比为 {ratio:.4f}，期望 2:1；"
                "该图可能不是标准等距柱状投影"
            )
            if self.config.strict_aspect:
                raise ValueError(message)
            warnings.warn(message, UserWarning, stacklevel=2)
            warning_messages.append(message)

        target_dir.mkdir(parents=True, exist_ok=True)
        views: list[PanoramaViewMetadata] = []

        overview_path = target_dir / "overview.jpg"
        overview = cv2.resize(
            panorama,
            (self.config.overview_width, self.config.overview_height),
            interpolation=cv2.INTER_AREA,
        )
        _write_jpeg(overview_path, overview, self.config.jpeg_quality)
        views.append(
            PanoramaViewMetadata(
                view_id="overview",
                source_image_path=str(source),
                source_image_id=image_id,
                width=self.config.overview_width,
                height=self.config.overview_height,
                output_path=str(overview_path),
                is_overview=True,
            )
        )

        projections: list[tuple[str, float, float]] = [
            (f"yaw_{self._yaw_label(yaw)}", yaw, 0.0)
            for yaw in self.config.horizontal_yaws
        ]
        # 迭代中：默认生产配置固定为 False，当前 Task 2/3 暂未调用；
        # 投影代码保留，供后续迭代或显式构造 PanoramaProjectionConfig 的几何测试使用。
        if self.config.include_downward:
            projections.extend(
                (
                    f"down_yaw_{self._yaw_label(yaw)}",
                    yaw,
                    self.config.downward_pitch,
                )
                for yaw in self.config.horizontal_yaws
            )

        for view_id, yaw, pitch in projections:
            output_path = target_dir / f"{view_id}.jpg"
            perspective = equirectangular_to_perspective(
                panorama,
                yaw=yaw,
                pitch=pitch,
                fov=self.config.fov,
                width=self.config.perspective_width,
                height=self.config.perspective_height,
            )
            _write_jpeg(output_path, perspective, self.config.jpeg_quality)
            views.append(
                PanoramaViewMetadata(
                    view_id=view_id,
                    source_image_path=str(source),
                    source_image_id=image_id,
                    yaw=float(yaw) % 360.0,
                    pitch=float(pitch),
                    fov=float(self.config.fov),
                    width=self.config.perspective_width,
                    height=self.config.perspective_height,
                    output_path=str(output_path),
                    is_overview=False,
                )
            )

        result = PanoramaViewSet(
            source_image_path=str(source),
            source_image_id=image_id,
            source_sha256=source_sha256,
            source_width=source_width,
            source_height=source_height,
            views=views,
            warnings=warning_messages,
            cache_hit=False,
        )
        payload = result.model_dump()
        payload["config_signature"] = self._config_signature()
        metadata_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return result


__all__ = [
    "PanoramaProjectionConfig",
    "PanoramaViewGenerator",
    "equirectangular_to_perspective",
]
