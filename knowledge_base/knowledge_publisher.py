"""将通过质量门禁的 DeepSeek 草稿发布到正式 RAG 知识目录。"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import MORPH_KEYS, RAG_PUBLISHED_KNOWLEDGE_DIR


_LINE_ID_RE = re.compile(r"^P\d{4}-L\d{4}$")


@dataclass(frozen=True)
class PublishPolicy:
    # 与 knowledge_curator 的 program_validated 门禁保持一致。
    min_confidence: float = 0.75
    min_fuzzy_score: float = 0.85


@dataclass(frozen=True)
class PublicationResult:
    published_records: list[dict[str, Any]]
    held_records: list[dict[str, Any]]
    output_paths: list[Path]


def hard_gate_errors(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if record.get("validation_errors"):
        errors.append("存在校验错误")
    illegal_metrics = sorted(set(record.get("metric_links") or []) - set(MORPH_KEYS))
    if illegal_metrics:
        errors.append(f"包含非法形态指标: {', '.join(illegal_metrics)}")
    evidence = list(record.get("evidence") or [])
    if not evidence:
        errors.append("没有证据")
    for index, item in enumerate(evidence, start=1):
        line_ids = list(item.get("line_ids") or [])
        if not line_ids:
            errors.append(f"证据 {index} 没有 OCR 行号")
        elif any(not _LINE_ID_RE.match(str(line_id)) for line_id in line_ids):
            errors.append(f"证据 {index} 含无效 OCR 行号")
        if item.get("match_type") not in {"exact", "fuzzy_ocr"}:
            errors.append(f"证据 {index} 无法定位")
    return errors


def soft_gate_errors(record: dict[str, Any], policy: PublishPolicy) -> list[str]:
    errors: list[str] = []
    if record.get("review_status") != "program_validated":
        errors.append("记录未通过程序校验")
    if float(record.get("confidence") or 0.0) < policy.min_confidence:
        errors.append("模型置信度不足")
    if record.get("needs_review"):
        errors.append("模型主动要求复核")
    if record.get("high_risk_normative_text"):
        errors.append("含强制性或禁止性规范措辞")
    for index, item in enumerate(record.get("evidence") or [], start=1):
        if (
            item.get("match_type") == "fuzzy_ocr"
            and float(item.get("match_score") or 0.0) < policy.min_fuzzy_score
        ):
            errors.append(f"证据 {index} OCR 匹配分数不足")
    return errors


def collect_publication_records(
    draft_dir: str | Path,
    *,
    include: list[str] | None = None,
    policy: PublishPolicy = PublishPolicy(),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(draft_dir).resolve()
    include_values = [value.casefold() for value in (include or [])]
    published: dict[str, dict[str, Any]] = {}
    held: list[dict[str, Any]] = []
    for path in sorted(root.rglob("batch-*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("schema_version") != "ai4city-knowledge-draft-v2":
            continue
        source_name = Path(str(document.get("source") or "")).name
        if include_values and not any(value in source_name.casefold() for value in include_values):
            continue
        source_path = Path(str(document.get("source") or ""))
        source_error = ""
        if not source_path.is_file():
            source_error = "源文档不存在"
        else:
            stat = source_path.stat()
            fingerprint = hashlib.sha256(
                f"{source_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
            ).hexdigest()
            if fingerprint != str(document.get("source_fingerprint") or ""):
                source_error = "源文档指纹已变化"
        for raw_record in document.get("records") or []:
            record = dict(raw_record)
            knowledge_id = str(record.get("knowledge_id") or "")
            hard_errors = hard_gate_errors(record)
            if source_error:
                hard_errors.append(source_error)
            soft_errors = soft_gate_errors(record, policy)
            if hard_errors or soft_errors:
                held.append(
                    {
                        "knowledge_id": knowledge_id,
                        "source": record.get("source"),
                        "title": record.get("title"),
                        "hard_gate_errors": hard_errors,
                        "soft_gate_errors": soft_errors,
                    }
                )
                continue
            record["publication"] = {
                "approval_type": "program_gate",
                "published_at": datetime.now(timezone.utc).isoformat(),
                "policy": {
                    "min_confidence": policy.min_confidence,
                    "min_fuzzy_score": policy.min_fuzzy_score,
                },
            }
            published[knowledge_id] = record
    return list(published.values()), held


def _safe_name(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    stem = "".join(character if character.isalnum() else "_" for character in value)
    stem = stem.strip("_")[:70] or "knowledge"
    return f"{stem}-{digest}.json"


def write_publication(
    records: list[dict[str, Any]],
    *,
    output_dir: str | Path = RAG_PUBLISHED_KNOWLEDGE_DIR,
) -> list[Path]:
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record.get("source") or "unknown"), []).append(record)
    paths: list[Path] = []
    for source, source_records in grouped.items():
        source_name = Path(source).name
        path = root / _safe_name(source_name)
        portable_records = []
        for record in source_records:
            portable = dict(record)
            # 正式知识会随仓库共享，只保留逻辑来源名；绝对路径仅用于发布前
            # 的源文件指纹校验和本地诊断报告，不写入可提交的知识 JSON。
            portable["source"] = source_name
            portable_records.append(portable)
        document = {
            "schema_version": "ai4city-rag-knowledge-v1",
            "source": source_name,
            "record_count": len(portable_records),
            "records": sorted(portable_records, key=lambda item: item["knowledge_id"]),
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)
        paths.append(path)
    return paths


__all__ = [
    "PublicationResult",
    "PublishPolicy",
    "collect_publication_records",
    "hard_gate_errors",
    "soft_gate_errors",
    "write_publication",
]
