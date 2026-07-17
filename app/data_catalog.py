"""发现后端维护的全景图和项目指标大表。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from app.metrics_table_loader import MetricsTable, MetricsTableError, list_workbook_sheets
from config import (
    METRICS_TABLE_DIR,
    METRICS_TABLE_PATH,
    PANORAMA_DIR,
    SCENE_MANIFEST_PATH,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
TABLE_SUFFIXES = {".xlsx", ".xlsm", ".csv"}


@dataclass(frozen=True)
class CatalogItem:
    label: str
    path: str

    def as_choice(self) -> tuple[str, str]:
        return self.label, self.path


@dataclass(frozen=True)
class BackendCatalog:
    images: tuple[CatalogItem, ...]
    metric_tables: tuple[CatalogItem, ...]


def _safe_resolve(path: str | Path, base: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def discover_images(
    image_dir: str | Path = PANORAMA_DIR,
    manifest_path: str | Path = SCENE_MANIFEST_PATH,
) -> tuple[CatalogItem, ...]:
    """优先读取 scenes.csv；没有清单时再扫描后端图片目录。"""

    root = Path(image_dir).resolve()
    manifest = Path(manifest_path).resolve()
    found: dict[str, CatalogItem] = {}
    if manifest.is_file():
        try:
            with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    raw_path = str(row.get("image_path") or row.get("图片路径") or "").strip()
                    if not raw_path:
                        continue
                    path = _safe_resolve(raw_path, manifest.parent)
                    if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                        continue
                    scene_id = str(row.get("scene_id") or row.get("场景编号") or "").strip()
                    space_type = str(row.get("space_type") or row.get("空间类型") or "").strip()
                    parts = [item for item in (scene_id, path.name, space_type) if item]
                    found[str(path)] = CatalogItem(" · ".join(parts), str(path))
        except (OSError, csv.Error):
            # 清单损坏时仍允许通过目录扫描使用后端图片。
            found = {}

    if not found and root.is_dir():
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                relative = path.relative_to(root)
                found[str(path.resolve())] = CatalogItem(str(relative), str(path.resolve()))
    return tuple(sorted(found.values(), key=lambda item: item.label.casefold()))


def _has_valid_metrics_sheet(path: Path) -> bool:
    try:
        for sheet in list_workbook_sheets(path):
            try:
                MetricsTable.from_file(path, sheet_name=sheet)
                return True
            except MetricsTableError:
                continue
    except (OSError, MetricsTableError):
        return False
    return False


def discover_metric_tables(
    table_dir: str | Path = METRICS_TABLE_DIR,
    explicit_path: str | Path | None = METRICS_TABLE_PATH,
) -> tuple[CatalogItem, ...]:
    """只返回至少有一个工作表满足七项指标列契约的后端大表。"""

    root = Path(table_dir).resolve()
    candidates: set[Path] = set()
    if explicit_path:
        explicit = Path(explicit_path).resolve()
        if explicit.is_file():
            candidates.add(explicit)
    if root.is_dir():
        candidates.update(
            path.resolve()
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in TABLE_SUFFIXES
        )

    items = []
    for path in sorted(candidates, key=lambda item: str(item).casefold()):
        if not _has_valid_metrics_sheet(path):
            continue
        try:
            label = str(path.relative_to(root))
        except ValueError:
            label = path.name
        items.append(CatalogItem(label, str(path)))
    return tuple(items)


def scan_backend_catalog() -> BackendCatalog:
    return BackendCatalog(
        images=discover_images(),
        metric_tables=discover_metric_tables(),
    )


__all__ = [
    "BackendCatalog",
    "CatalogItem",
    "discover_images",
    "discover_metric_tables",
    "scan_backend_catalog",
]
