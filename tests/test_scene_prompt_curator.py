from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from knowledge_base.scene_prompt_curator import (
    ScenePromptCase,
    SourceParagraph,
    curate_scene_case,
    draft_path,
    file_sha256,
    parse_scene_prompt_docx,
    publish_scene_prompt_cases,
    validate_scene_case_payload,
    write_json_atomic,
)


WORD_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _paragraph(parent, text: str, style: str = "") -> None:
    paragraph = ElementTree.SubElement(parent, f"{{{WORD_NAMESPACE}}}p")
    if style:
        properties = ElementTree.SubElement(
            paragraph, f"{{{WORD_NAMESPACE}}}pPr"
        )
        style_node = ElementTree.SubElement(
            properties, f"{{{WORD_NAMESPACE}}}pStyle"
        )
        style_node.set(f"{{{WORD_NAMESPACE}}}val", style)
    run = ElementTree.SubElement(paragraph, f"{{{WORD_NAMESPACE}}}r")
    text_node = ElementTree.SubElement(run, f"{{{WORD_NAMESPACE}}}t")
    text_node.text = text


def _write_fixture_docx(path: Path) -> None:
    document = ElementTree.Element(f"{{{WORD_NAMESPACE}}}document")
    body = ElementTree.SubElement(document, f"{{{WORD_NAMESPACE}}}body")
    labels = (("蓝绿", "BLUE"), ("商办", "OFFICE"), ("社区", "COMMUNITY"))
    for label, image_prefix in labels:
        _paragraph(body, label, "2")
        for ordinal in range(1, 11):
            _paragraph(body, f"{ordinal}、VID_{image_prefix}_{ordinal:02d}", "3")
            _paragraph(body, "【场景属性】")
            _paragraph(body, f"{label}案例原文明确要求保留道路并增加座椅。")
            _paragraph(body, "输出规格：4096×2048，2:1等距柱状投影360°全景图。")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "word/document.xml",
            ElementTree.tostring(document, encoding="utf-8", xml_declaration=True),
        )


def _case() -> ScenePromptCase:
    paragraphs = (
        SourceParagraph("P0001", 1, "1、VID_TEST_001", "3"),
        SourceParagraph(
            "P0002", 2, "社区案例原文明确要求保留道路并增加座椅。", ""
        ),
        SourceParagraph(
            "P0003", 3, "输出规格：4096×2048，2:1等距柱状投影360°全景图。", ""
        ),
    )
    return ScenePromptCase(
        scene_type="community",
        scene_label="社区",
        ordinal=1,
        title="1、VID_TEST_001",
        image_id="VID_TEST_001",
        paragraphs=paragraphs,
    )


def _evidence_item(text: str = "保留道路") -> dict:
    return {
        "text": text,
        "evidence_paragraph_ids": ["P0002"],
        "evidence_quote": "明确要求保留道路",
    }


def _valid_payload() -> dict:
    return {
        "scene_attributes": {
            "summary": "社区案例",
            "evidence_paragraph_ids": ["P0002"],
            "evidence_quote": "社区案例原文明确",
        },
        "experience_diagnosis": [_evidence_item()],
        "current_strengths": [],
        "problems": [],
        "update_goals": [_evidence_item("增加可停留设施")],
        "object_actions": [
            {
                "action": "add",
                "object_type": "street_furniture (seating)",
                "position": "道路旁",
                "quantity": "未说明",
                "attributes": ["不阻碍道路"],
                "rationale": "增加停留条件",
                "evidence_paragraph_ids": ["P0002"],
                "evidence_quote": "保留道路并增加座椅",
            }
        ],
        "spatial_relations": [_evidence_item()],
        "unchanged_regions": [_evidence_item()],
        "constraints": [_evidence_item()],
        "visual_requirements": [_evidence_item()],
        "prohibited_actions": [_evidence_item()],
        "output_spec": {
            "width": 4096,
            "height": 2048,
            "projection": "2:1等距柱状投影360°全景图",
            "style": "写实",
            "evidence_paragraph_ids": ["P0003"],
            "evidence_quote": "4096×2048，2:1等距柱状投影360°全景图",
        },
        "confidence": 0.92,
        "needs_review": False,
        "review_notes": [],
    }


class _FakeClient:
    flash_model = "flash"
    pro_model = "pro"

    def __init__(self, flash_payload: dict, pro_payload: dict | None = None) -> None:
        self.flash_payload = flash_payload
        self.pro_payload = pro_payload or flash_payload
        self.calls: list[bool] = []

    def complete_json(self, system: str, user: str, *, use_pro: bool = False):
        self.calls.append(use_pro)
        payload = self.pro_payload if use_pro else self.flash_payload
        return ("pro" if use_pro else "flash", json.dumps(payload, ensure_ascii=False))


class ScenePromptCuratorTests(unittest.TestCase):
    def setUp(self) -> None:
        test_root = Path(__file__).resolve().parents[1] / "outputs" / "test_temp"
        test_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=test_root)
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_docx_parser_finds_ten_cases_per_scene(self) -> None:
        source = self.root / "prompt.docx"
        _write_fixture_docx(source)
        cases = parse_scene_prompt_docx(source)
        self.assertEqual(len(cases), 30)
        self.assertEqual(
            {scene: sum(case.scene_type == scene for case in cases) for scene in {
                "blue_green", "commercial_office", "community"
            }},
            {"blue_green": 10, "commercial_office": 10, "community": 10},
        )
        self.assertEqual(cases[0].image_id, "VID_BLUE_01")
        self.assertEqual(cases[-1].knowledge_id, "scene-prompt-community-10")

    def test_validation_rejects_unlocatable_quote_and_wrong_dimensions(self) -> None:
        payload = _valid_payload()
        payload["object_actions"][0]["evidence_quote"] = "原文中不存在的连续证据"
        payload["output_spec"]["width"] = 2048
        errors = validate_scene_case_payload(payload, _case())
        self.assertTrue(any("无法在所引段落" in error for error in errors))
        self.assertTrue(any("4096×2048" in error for error in errors))

    def test_invalid_flash_is_repaired_by_pro(self) -> None:
        invalid = _valid_payload()
        invalid["confidence"] = 0.5
        client = _FakeClient(invalid, _valid_payload())
        draft = curate_scene_case(
            client,
            _case(),
            source_name="prompt.docx",
            source_fingerprint="fingerprint",
            auto_pro=True,
        )
        self.assertEqual(client.calls, [False, True])
        self.assertEqual(draft["review_status"], "program_validated")
        self.assertEqual(draft["record"]["curation"]["tier"], "pro")

    def test_publish_requires_every_case_to_be_program_validated(self) -> None:
        source = self.root / "prompt.docx"
        _write_fixture_docx(source)
        cases = parse_scene_prompt_docx(source)
        fingerprint = file_sha256(source)
        draft_dir = self.root / "drafts"
        published_path = self.root / "published" / "scene_prompt_examples.json"
        first = cases[0]
        payload = _valid_payload()
        # Fixture case paragraph IDs differ from the compact validation fixture, so
        # publication is tested with an already valid record wrapper.
        wrapper = {
            "schema_version": "ai4city-scene-prompt-draft-v1",
            "source_fingerprint": fingerprint,
            "case_source_hash": first.source_hash,
            "knowledge_id": first.knowledge_id,
            "review_status": "program_validated",
            "validation_errors": [],
            "record": {"knowledge_id": first.knowledge_id, "scene_type": first.scene_type},
        }
        write_json_atomic(draft_path(draft_dir, first), wrapper)
        published, missing = publish_scene_prompt_cases(
            cases,
            draft_dir=draft_dir,
            output_path=published_path,
            source_name=source.name,
            source_fingerprint=fingerprint,
        )
        self.assertIsNone(published)
        self.assertEqual(len(missing), 29)
        self.assertFalse(published_path.exists())


if __name__ == "__main__":
    unittest.main()
