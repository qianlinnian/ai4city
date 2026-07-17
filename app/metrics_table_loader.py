"""Read the seven morphology metrics from a project data table.

The online workflow consumes precomputed Task 1 results instead of running the
image model.  This module intentionally has no pandas/openpyxl dependency: it
reads CSV files with :mod:`csv` and modern Excel files with the Python standard
library (``zipfile`` + ``xml.etree``).
"""

from __future__ import annotations

import csv
import math
import posixpath
import re
import zipfile
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Mapping, Sequence
from xml.etree import ElementTree as ET

from config import MORPH_BOUNDS, MORPH_KEYS


CSV_SHEET_NAME = "CSV"

IMAGE_COLUMN_ALIASES = (
    "图像名称",
    "图片名称",
    "图像名",
    "图片名",
    "文件名称",
    "文件名",
    "照片名称",
    "影像名称",
    "全景图名称",
    "原始图片名称",
    "image_name",
    "image",
    "filename",
    "file_name",
)

# The public table contract deliberately does not use the historical "综合-"
# prefix.  Parenthesised abbreviations are accepted for convenient migration.
MORPH_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "green_view": ("绿视率", "绿视率(GVI)", "绿视率(%)", "GVI"),
    "blue_view": ("蓝视率", "蓝视率(BVI)", "蓝视率(%)", "BVI"),
    "sky_view": ("天空可视率", "天空可视率(SVF)", "天空可视率(%)", "SVF"),
    "built_ratio": ("人造物占比", "人造物占比(BR)", "人造物占比(%)", "BR"),
    "color_richness": ("色彩丰富度", "色彩丰富度(CR)", "CR"),
    "edge_density": ("边缘密度", "边缘密度(ED)", "边缘密度(%)", "ED"),
    "skyline_variance": ("天际线变化率", "天际线变化率(SVR)", "天际线变化率(%)", "SVR"),
}

MORPH_LABELS = {
    "green_view": "绿视率",
    "blue_view": "蓝视率",
    "sky_view": "天空可视率",
    "built_ratio": "人造物占比",
    "color_richness": "色彩丰富度",
    "edge_density": "边缘密度",
    "skyline_variance": "天际线变化率",
}

RATIO_KEYS = frozenset(MORPH_KEYS) - {"color_richness"}
_NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_NS_REL_DOC = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_NS_REL_PACKAGE = "http://schemas.openxmlformats.org/package/2006/relationships"
_CELL_REF_RE = re.compile(r"^([A-Z]+)", re.IGNORECASE)


class MetricsTableError(ValueError):
    """Base class for table input errors that are safe to show in the UI."""


class UnsupportedTableError(MetricsTableError):
    pass


class SheetNotFoundError(MetricsTableError):
    pass


class MissingColumnsError(MetricsTableError):
    def __init__(self, missing: Sequence[str]) -> None:
        self.missing = tuple(missing)
        super().__init__("数据表缺少必需列：" + "、".join(self.missing))


class InvalidMetricValueError(MetricsTableError):
    def __init__(self, key: str, value: Any, reason: str, row_index: int | None = None) -> None:
        self.key = key
        self.value = value
        self.row_index = row_index
        location = f"第 {row_index} 行" if row_index is not None else "数据表"
        super().__init__(f"{location}的{MORPH_LABELS[key]}无效（{value!r}）：{reason}")


@dataclass(frozen=True)
class CandidateRow:
    row_index: int
    image_name: str
    score: float = 1.0


@dataclass(frozen=True)
class MetricMatch:
    """Result of matching an uploaded image to a row in the large table."""

    query: str
    status: str
    match_type: str | None = None
    row_index: int | None = None
    image_name: str | None = None
    metrics: dict[str, float] | None = None
    candidates: tuple[CandidateRow, ...] = field(default_factory=tuple)
    error: str | None = None

    @property
    def matched(self) -> bool:
        return self.status == "matched" and self.metrics is not None


@dataclass(frozen=True)
class MetricComparison:
    key: str
    label: str
    before: float
    after: float
    delta: float
    direction: str
    target: float | None = None
    target_deviation: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "before": self.before,
            "after": self.after,
            "delta": self.delta,
            "direction": self.direction,
            "target": self.target,
            "target_deviation": self.target_deviation,
        }


@dataclass(frozen=True)
class _RawRow:
    row_index: int
    values: tuple[Any, ...]


def _normalise_header(value: Any) -> str:
    text = str(value or "").strip().replace("（", "(").replace("）", ")")
    return re.sub(r"[\s\-_]+", "", text).casefold()


def _normalise_filename(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    return text.rsplit("/", 1)[-1].casefold()


def _filename_stem(value: Any) -> str:
    name = _normalise_filename(value)
    return name.rsplit(".", 1)[0] if "." in name else name


def _column_number(cell_ref: str | None, fallback: int) -> int:
    match = _CELL_REF_RE.match(cell_ref or "")
    if not match:
        return fallback
    number = 0
    for char in match.group(1).upper():
        number = number * 26 + ord(char) - ord("A") + 1
    return number - 1


def _xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return ["".join(node.text or "" for node in item.iter(f"{{{_NS_MAIN}}}t")) for item in root]


def _xlsx_sheet_parts(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    try:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    except (KeyError, ET.ParseError) as exc:
        raise UnsupportedTableError("Excel 文件结构损坏或不是有效的 .xlsx 文件") from exc

    targets = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in relationships.findall(f"{{{_NS_REL_PACKAGE}}}Relationship")
        if "Id" in rel.attrib and "Target" in rel.attrib
    }
    parts: list[tuple[str, str]] = []
    for sheet in workbook.findall(f".//{{{_NS_MAIN}}}sheet"):
        name = sheet.attrib.get("name", "")
        rel_id = sheet.attrib.get(f"{{{_NS_REL_DOC}}}id", "")
        target = targets.get(rel_id)
        if not target:
            continue
        if target.startswith("/"):
            part = target.lstrip("/")
        else:
            part = posixpath.normpath(posixpath.join("xl", target))
        parts.append((name, part))
    return parts


def _xlsx_cell_value(cell: ET.Element, shared_strings: Sequence[str]) -> Any:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.iter(f"{{{_NS_MAIN}}}t"))
    value_node = cell.find(f"{{{_NS_MAIN}}}v")
    if value_node is None or value_node.text is None:
        return None
    raw = value_node.text
    if cell_type == "s":
        try:
            return shared_strings[int(raw)]
        except (ValueError, IndexError) as exc:
            raise UnsupportedTableError("Excel sharedStrings 索引无效") from exc
    if cell_type in {"str", "e"}:
        return raw
    if cell_type == "b":
        return raw == "1"
    try:
        number = float(raw)
        return int(number) if number.is_integer() else number
    except ValueError:
        return raw


def _read_xlsx_rows(path: Path, sheet_name: str | None) -> tuple[str, list[_RawRow]]:
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise UnsupportedTableError(f"无法读取 Excel 文件：{path.name}") from exc
    with archive:
        sheets = _xlsx_sheet_parts(archive)
        if not sheets:
            raise UnsupportedTableError("Excel 文件中没有可读取的工作表")
        selected = next((item for item in sheets if item[0] == sheet_name), None) if sheet_name else sheets[0]
        if selected is None:
            raise SheetNotFoundError(f"Excel 中不存在工作表：{sheet_name}")
        name, part = selected
        try:
            root = ET.fromstring(archive.read(part))
        except (KeyError, ET.ParseError) as exc:
            raise UnsupportedTableError(f"无法读取工作表：{name}") from exc
        shared_strings = _xlsx_shared_strings(archive)
        rows: list[_RawRow] = []
        for fallback_row, row in enumerate(root.findall(f".//{{{_NS_MAIN}}}sheetData/{{{_NS_MAIN}}}row"), 1):
            row_index = int(row.attrib.get("r", fallback_row))
            cells: dict[int, Any] = {}
            next_column = 0
            for cell in row.findall(f"{{{_NS_MAIN}}}c"):
                column = _column_number(cell.attrib.get("r"), next_column)
                cells[column] = _xlsx_cell_value(cell, shared_strings)
                next_column = column + 1
            width = max(cells, default=-1) + 1
            rows.append(_RawRow(row_index, tuple(cells.get(index) for index in range(width))))
        return name, rows


def _read_csv_rows(path: Path) -> tuple[str, list[_RawRow]]:
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                return CSV_SHEET_NAME, [
                    _RawRow(index, tuple(row)) for index, row in enumerate(csv.reader(handle), 1)
                ]
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnsupportedTableError(f"无法识别 CSV 文件编码：{path.name}") from last_error


def list_workbook_sheets(path: str | Path) -> list[str]:
    """Return available sheet names without importing an Excel dependency."""

    table_path = Path(path)
    suffix = table_path.suffix.casefold()
    if suffix == ".csv":
        return [CSV_SHEET_NAME]
    if suffix not in {".xlsx", ".xlsm"}:
        raise UnsupportedTableError("仅支持 .xlsx、.xlsm 和 .csv 数据表")
    try:
        with zipfile.ZipFile(table_path) as archive:
            return [name for name, _ in _xlsx_sheet_parts(archive)]
    except (OSError, zipfile.BadZipFile) as exc:
        raise UnsupportedTableError(f"无法读取 Excel 文件：{table_path.name}") from exc


def resolve_required_columns(headers: Sequence[Any]) -> tuple[str, dict[str, str]]:
    """Resolve the image column and seven metric columns from table headers."""

    by_normalised: dict[str, str] = {}
    for header in headers:
        if header is not None and str(header).strip():
            by_normalised.setdefault(_normalise_header(header), str(header).strip())

    image_column = next(
        (by_normalised[_normalise_header(alias)] for alias in IMAGE_COLUMN_ALIASES if _normalise_header(alias) in by_normalised),
        None,
    )
    columns: dict[str, str] = {}
    for key in MORPH_KEYS:
        for alias in MORPH_COLUMN_ALIASES[key]:
            actual = by_normalised.get(_normalise_header(alias))
            if actual is not None:
                columns[key] = actual
                break

    missing: list[str] = []
    if image_column is None:
        missing.append("图像名称（或图片名称）")
    missing.extend(MORPH_LABELS[key] for key in MORPH_KEYS if key not in columns)
    if missing:
        raise MissingColumnsError(missing)
    return image_column, columns


def convert_metric_value(key: str, value: Any, row_index: int | None = None) -> float:
    """Convert a number/percentage string and enforce ``MORPH_BOUNDS``."""

    if key not in MORPH_BOUNDS:
        raise KeyError(key)
    if value is None or isinstance(value, bool) or (isinstance(value, str) and not value.strip()):
        raise InvalidMetricValueError(key, value, "值为空或不是数值", row_index)

    is_percent = False
    raw = value
    if isinstance(value, str):
        text = value.strip().replace(",", "").replace("％", "%")
        is_percent = text.endswith("%")
        if is_percent:
            text = text[:-1].strip()
        try:
            raw = float(text)
        except ValueError as exc:
            raise InvalidMetricValueError(key, value, "无法转换为数值", row_index) from exc
    try:
        number = float(raw)
    except (TypeError, ValueError) as exc:
        raise InvalidMetricValueError(key, value, "无法转换为数值", row_index) from exc

    if not math.isfinite(number):
        raise InvalidMetricValueError(key, value, "必须是有限数值", row_index)
    if is_percent:
        if key not in RATIO_KEYS:
            raise InvalidMetricValueError(key, value, "色彩丰富度不接受百分数", row_index)
        number /= 100.0

    lower, upper = MORPH_BOUNDS[key]
    if not lower <= number <= upper:
        raise InvalidMetricValueError(key, value, f"应位于 {lower:g}～{upper:g} 之间", row_index)
    return number


def validate_metrics(metrics: Mapping[str, Any], row_index: int | None = None) -> dict[str, float]:
    missing = [MORPH_LABELS[key] for key in MORPH_KEYS if key not in metrics]
    if missing:
        raise MissingColumnsError(missing)
    return {key: convert_metric_value(key, metrics[key], row_index) for key in MORPH_KEYS}


class MetricsTable:
    """A selected worksheet with validated column mapping and row matching."""

    def __init__(
        self,
        path: Path,
        sheet_name: str,
        headers: Sequence[Any],
        rows: Sequence[_RawRow],
        header_row_index: int,
    ) -> None:
        self.path = path
        self.sheet_name = sheet_name
        self.headers = tuple(str(value or "").strip() for value in headers)
        self.image_column, self.metric_columns = resolve_required_columns(self.headers)
        self.header_row_index = header_row_index
        self._rows = tuple(rows)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        sheet_name: str | None = None,
        header_row: int | None = None,
    ) -> "MetricsTable":
        """Load a CSV/XLSX table; auto-detect the header within the first 50 rows."""

        table_path = Path(path)
        suffix = table_path.suffix.casefold()
        if suffix == ".csv":
            selected_sheet, raw_rows = _read_csv_rows(table_path)
            if sheet_name not in {None, CSV_SHEET_NAME}:
                raise SheetNotFoundError(f"CSV 仅包含虚拟工作表：{CSV_SHEET_NAME}")
        elif suffix in {".xlsx", ".xlsm"}:
            selected_sheet, raw_rows = _read_xlsx_rows(table_path, sheet_name)
        else:
            raise UnsupportedTableError("仅支持 .xlsx、.xlsm 和 .csv 数据表")
        if not raw_rows:
            raise MissingColumnsError(["表头和数据"])

        if header_row is not None:
            header = next((row for row in raw_rows if row.row_index == header_row), None)
            if header is None:
                raise MetricsTableError(f"数据表中不存在第 {header_row} 行")
        else:
            header = None
            best_missing = 999
            for row in raw_rows[:50]:
                try:
                    resolve_required_columns(row.values)
                    header = row
                    break
                except MissingColumnsError as exc:
                    best_missing = min(best_missing, len(exc.missing))
            if header is None:
                # Let the first row produce the detailed, user-facing list.
                resolve_required_columns(raw_rows[0].values)
                raise MetricsTableError(f"前 50 行未找到完整表头（最少仍缺 {best_missing} 列）")

        data_rows = [row for row in raw_rows if row.row_index > header.row_index and any(v not in (None, "") for v in row.values)]
        return cls(table_path, selected_sheet, header.values, data_rows, header.row_index)

    def _row_dict(self, row: _RawRow) -> dict[str, Any]:
        return {
            header: row.values[index] if index < len(row.values) else None
            for index, header in enumerate(self.headers)
            if header
        }

    def row_to_metrics(self, row: _RawRow | Mapping[str, Any]) -> dict[str, float]:
        values = self._row_dict(row) if isinstance(row, _RawRow) else row
        row_index = row.row_index if isinstance(row, _RawRow) else None
        return {
            key: convert_metric_value(key, values.get(column), row_index)
            for key, column in self.metric_columns.items()
        }

    @property
    def image_names(self) -> tuple[str, ...]:
        return tuple(str(self._row_dict(row).get(self.image_column) or "").strip() for row in self._rows)

    def metrics_for_row(self, row_index: int) -> dict[str, float]:
        """Return validated metrics for a user-selected candidate row."""

        row = next((item for item in self._rows if item.row_index == row_index), None)
        if row is None:
            raise MetricsTableError(f"数据表中不存在第 {row_index} 行指标数据")
        return self.row_to_metrics(row)

    def _candidate_rows(self, query: str, limit: int) -> tuple[CandidateRow, ...]:
        query_stem = _filename_stem(query)
        candidates: list[CandidateRow] = []
        for row in self._rows:
            image_name = str(self._row_dict(row).get(self.image_column) or "").strip()
            if not image_name:
                continue
            score = SequenceMatcher(None, query_stem, _filename_stem(image_name)).ratio()
            candidates.append(CandidateRow(row.row_index, image_name, round(score, 4)))
        candidates.sort(key=lambda item: (-item.score, item.row_index))
        return tuple(candidates[:limit])

    def match_image(self, image_name: str | Path, candidate_limit: int = 5) -> MetricMatch:
        """Match by full filename first, then stem; never silently choose a fuzzy row."""

        query = str(image_name)
        normalised = _normalise_filename(query)
        stem = _filename_stem(query)
        if not normalised:
            return MetricMatch(query, "not_found", candidates=self._candidate_rows(query, candidate_limit))

        exact: list[_RawRow] = []
        stem_matches: list[_RawRow] = []
        for row in self._rows:
            candidate = self._row_dict(row).get(self.image_column)
            if _normalise_filename(candidate) == normalised:
                exact.append(row)
            elif _filename_stem(candidate) == stem:
                stem_matches.append(row)
        matches, match_type = (exact, "exact") if exact else (stem_matches, "stem")
        if not matches:
            return MetricMatch(query, "not_found", candidates=self._candidate_rows(query, candidate_limit))
        if len(matches) > 1:
            candidates = tuple(
                CandidateRow(row.row_index, str(self._row_dict(row).get(self.image_column) or "")) for row in matches
            )
            return MetricMatch(query, "ambiguous", match_type=match_type, candidates=candidates)

        row = matches[0]
        values = self._row_dict(row)
        matched_name = str(values.get(self.image_column) or "").strip()
        try:
            metrics = self.row_to_metrics(row)
        except MetricsTableError as exc:
            return MetricMatch(
                query,
                "invalid",
                match_type=match_type,
                row_index=row.row_index,
                image_name=matched_name,
                error=str(exc),
            )
        return MetricMatch(
            query,
            "matched",
            match_type=match_type,
            row_index=row.row_index,
            image_name=matched_name,
            metrics=metrics,
        )

    def validate_rows(self) -> list[str]:
        """Return all row-level errors without aborting at the first bad row."""

        errors: list[str] = []
        for row in self._rows:
            try:
                self.row_to_metrics(row)
            except MetricsTableError as exc:
                errors.append(str(exc))
        return errors


def compare_metrics(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    target: Mapping[str, Any] | None = None,
    tolerance: float = 1e-9,
) -> list[MetricComparison]:
    """Build ordered before/after comparisons for UI tables and persistence."""

    before_values = validate_metrics(before)
    after_values = validate_metrics(after)
    target_values = validate_metrics(target) if target is not None else None
    comparisons: list[MetricComparison] = []
    for key in MORPH_KEYS:
        delta = after_values[key] - before_values[key]
        direction = "保持"
        if delta > tolerance:
            direction = "增加"
        elif delta < -tolerance:
            direction = "降低"
        target_value = target_values[key] if target_values else None
        comparisons.append(
            MetricComparison(
                key=key,
                label=MORPH_LABELS[key],
                before=before_values[key],
                after=after_values[key],
                delta=delta,
                direction=direction,
                target=target_value,
                target_deviation=after_values[key] - target_value if target_value is not None else None,
            )
        )
    return comparisons


__all__ = [
    "CSV_SHEET_NAME",
    "IMAGE_COLUMN_ALIASES",
    "MORPH_COLUMN_ALIASES",
    "CandidateRow",
    "InvalidMetricValueError",
    "MetricComparison",
    "MetricMatch",
    "MetricsTable",
    "MetricsTableError",
    "MissingColumnsError",
    "SheetNotFoundError",
    "UnsupportedTableError",
    "compare_metrics",
    "convert_metric_value",
    "list_workbook_sheets",
    "resolve_required_columns",
    "validate_metrics",
]
