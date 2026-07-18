from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from app.metrics_table_loader import (
    MORPH_COLUMN_ALIASES,
    InvalidMetricValueError,
    MetricsTable,
    MissingColumnsError,
    compare_metrics,
    convert_metric_value,
    list_workbook_sheets,
)


HEADERS = [
    "项目编号",
    "图片名称",
    "绿视率(GVI)",
    "蓝视率",
    "天空可视率",
    "人造物占比(BR)",
    "色彩丰富度(CR)",
    "边缘密度",
    "天际线变化率(SVR)",
]


def _csv_text(*rows: list[object]) -> str:
    def escape(value: object) -> str:
        text = str(value)
        return '"' + text.replace('"', '""') + '"' if "," in text else text

    return "\n".join(",".join(escape(value) for value in row) for row in rows)


def _inline_cell(ref: str, value: str) -> str:
    return f'<c r="{ref}" t="inlineStr"><is><t>{value}</t></is></c>'


def _number_cell(ref: str, value: float) -> str:
    return f'<c r="{ref}"><v>{value}</v></c>'


def _create_xlsx(path: Path) -> None:
    main_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel_doc_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    workbook = f'''<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="{main_ns}" xmlns:r="{rel_doc_ns}"><sheets>
<sheet name="说明" sheetId="1" r:id="rId1"/>
<sheet name="数据总表" sheetId="2" r:id="rId2"/>
</sheets></workbook>'''
    relationships = '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Target="worksheets/sheet1.xml" Type="worksheet"/>
<Relationship Id="rId2" Target="/xl/worksheets/sheet2.xml" Type="worksheet"/>
</Relationships>'''
    sheet1 = f'''<worksheet xmlns="{main_ns}"><sheetData><row r="1">
{_inline_cell("A1", "说明文字")}</row></sheetData></worksheet>'''
    header_cells = "".join(
        _inline_cell(f"{chr(ord('A') + index)}2", header) for index, header in enumerate(HEADERS)
    )
    value_cells = [
        _inline_cell("A3", "P-01"),
        '<c r="B3" t="s"><v>0</v></c>',
        _number_cell("C3", 0.31),
        _number_cell("D3", 0.12),
        _number_cell("E3", 0.25),
        _number_cell("F3", 0.44),
        _number_cell("G3", 14),
        _number_cell("H3", 0.06),
        _number_cell("I3", 0.02),
    ]
    sheet2 = f'''<worksheet xmlns="{main_ns}"><sheetData>
<row r="1">{_inline_cell("A1", "某项目的大表")}</row>
<row r="2">{header_cells}</row>
<row r="3">{"".join(value_cells)}</row>
</sheetData></worksheet>'''
    shared_strings = f'''<sst xmlns="{main_ns}" count="1" uniqueCount="1">
<si><r><t>square</t></r><r><t>_01.jpg</t></r></si></sst>'''
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", relationships)
        archive.writestr("xl/worksheets/sheet1.xml", sheet1)
        archive.writestr("xl/worksheets/sheet2.xml", sheet2)
        archive.writestr("xl/sharedStrings.xml", shared_strings)


class MetricsTableLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        # Keep test artifacts inside the writable workspace on restricted CI.
        self.temp_dir = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_csv(self, *rows: list[object]) -> Path:
        path = self.root / "项目大表.csv"
        path.write_text(_csv_text(*rows), encoding="utf-8-sig")
        return path

    def test_main_column_contract_does_not_require_comprehensive_prefix(self) -> None:
        for aliases in MORPH_COLUMN_ALIASES.values():
            self.assertFalse(any(alias.startswith("综合-") for alias in aliases))

    def test_csv_exact_match_converts_percentage_strings(self) -> None:
        path = self.write_csv(
            HEADERS,
            ["P-01", "scene_01.jpg", "25%", 0.10, 0.20, 0.40, 12, "5％", 0.03],
        )
        table = MetricsTable.from_file(path)
        match = table.match_image(r"C:\uploads\SCENE_01.JPG")

        self.assertTrue(match.matched)
        self.assertEqual(match.match_type, "exact")
        self.assertEqual(match.row_index, 2)
        self.assertEqual(
            match.metrics,
            {
                "green_view": 0.25,
                "blue_view": 0.10,
                "sky_view": 0.20,
                "built_ratio": 0.40,
                "color_richness": 12.0,
                "edge_density": 0.05,
                "skyline_variance": 0.03,
            },
        )

    def test_stem_match_and_fuzzy_candidates_do_not_silently_select(self) -> None:
        path = self.write_csv(
            HEADERS,
            ["P-01", "plaza_north.jpg", 0.2, 0.1, 0.3, 0.4, 10, 0.05, 0.02],
            ["P-02", "park_south.jpg", 0.4, 0.1, 0.2, 0.2, 13, 0.04, 0.03],
        )
        table = MetricsTable.from_file(path)

        stem_match = table.match_image("plaza_north.png")
        self.assertTrue(stem_match.matched)
        self.assertEqual(stem_match.match_type, "stem")

        missing = table.match_image("plaza_nort.jpg")
        self.assertEqual(missing.status, "not_found")
        self.assertIsNone(missing.metrics)
        self.assertEqual(missing.candidates[0].image_name, "plaza_north.jpg")
        selected = table.metrics_for_row(missing.candidates[0].row_index)
        self.assertEqual(selected["green_view"], 0.2)

    def test_duplicate_stem_is_ambiguous_but_full_filename_has_priority(self) -> None:
        path = self.write_csv(
            HEADERS,
            ["P-01", "same.jpg", 0.2, 0.1, 0.3, 0.4, 10, 0.05, 0.02],
            ["P-02", "same.png", 0.3, 0.1, 0.3, 0.4, 11, 0.05, 0.02],
        )
        table = MetricsTable.from_file(path)

        self.assertEqual(table.match_image("same.jpg").status, "matched")
        ambiguous = table.match_image("same.webp")
        self.assertEqual(ambiguous.status, "ambiguous")
        self.assertEqual(len(ambiguous.candidates), 2)

    def test_missing_column_and_out_of_bounds_are_reported_in_chinese(self) -> None:
        missing_path = self.write_csv(
            HEADERS[:-1],
            ["P-01", "a.jpg", 0.2, 0.1, 0.3, 0.4, 10, 0.05],
        )
        with self.assertRaisesRegex(MissingColumnsError, "天际线变化率"):
            MetricsTable.from_file(missing_path)

        invalid_path = self.root / "invalid.csv"
        invalid_path.write_text(
            _csv_text(HEADERS, ["P-01", "a.jpg", "120%", 0.1, 0.3, 0.4, 10, 0.05, 0.02]),
            encoding="utf-8-sig",
        )
        invalid = MetricsTable.from_file(invalid_path).match_image("a.jpg")
        self.assertEqual(invalid.status, "invalid")
        self.assertIn("第 2 行", invalid.error or "")
        self.assertIn("0～1", invalid.error or "")

    def test_value_conversion_rejects_nonfinite_and_color_percentage(self) -> None:
        with self.assertRaises(InvalidMetricValueError):
            convert_metric_value("green_view", "NaN")
        with self.assertRaisesRegex(InvalidMetricValueError, "不接受百分数"):
            convert_metric_value("color_richness", "50%")

    def test_xlsx_lists_sheets_reads_shared_inline_and_numeric_cells(self) -> None:
        path = self.root / "项目大表.xlsx"
        _create_xlsx(path)

        self.assertEqual(list_workbook_sheets(path), ["说明", "数据总表"])
        table = MetricsTable.from_file(path, sheet_name="数据总表")
        self.assertEqual(table.header_row_index, 2)
        match = table.match_image("square_01.jpg")
        self.assertTrue(match.matched)
        self.assertEqual(match.metrics["green_view"], 0.31)
        self.assertEqual(match.metrics["color_richness"], 14.0)

    def test_compare_metrics_preserves_config_order_and_target_deviation(self) -> None:
        before = {
            "green_view": 0.2,
            "blue_view": 0.1,
            "sky_view": 0.3,
            "built_ratio": 0.4,
            "color_richness": 10,
            "edge_density": 0.05,
            "skyline_variance": 0.02,
        }
        after = {**before, "green_view": 0.3, "built_ratio": 0.35}
        target = {**before, "green_view": 0.32, "built_ratio": 0.34}

        result = compare_metrics(before, after, target)
        self.assertEqual([item.key for item in result], list(before))
        self.assertEqual(result[0].direction, "增加")
        self.assertAlmostEqual(result[0].target_deviation or 0, -0.02)
        self.assertEqual(result[3].direction, "降低")
        self.assertEqual(result[1].direction, "保持")


if __name__ == "__main__":
    unittest.main()
