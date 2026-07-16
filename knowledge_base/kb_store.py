"""
本地知识库
==========
存储：人工干预轨迹、提示词润色差异、指标偏差校正、Few-shot 检索。

文件位置: knowledge_base/data/memories.json
         knowledge_base/data/mapping_rules.json
"""

from __future__ import annotations

import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config import EXPERIENCE_KEYS, KB_DIR
from schemas.models import MemoryRecord


DEFAULT_MAPPING_RULES = {
    "description": "体验感受 → 形态要素 的经验映射（制图员默认规则，可被知识库记忆覆盖）",
    "rules": [
        {
            "when_high": "restoration",
            "adjust": {"green_view": +0.12, "sky_view": -0.04, "edge_density": -0.02},
            "layout_hint": "提高前景灌木与中景乔木围合，形成自然庇护感",
        },
        {
            "when_high": "comfort",
            "adjust": {"green_view": +0.08, "built_ratio": -0.06, "color_richness": +1.0},
            "layout_hint": "降低硬质界面压迫，增加柔和材质与遮阴",
        },
        {
            "when_high": "safety",
            "adjust": {"sky_view": +0.05, "edge_density": -0.03, "built_ratio": -0.04},
            "layout_hint": "打开视线通廊，减少死角与过密遮挡，提升界面整洁",
        },
        {
            "when_high": "pleasure",
            "adjust": {"color_richness": +2.0, "blue_view": +0.03, "green_view": +0.05},
            "layout_hint": "增加花色层次与水体/反射点缀，丰富视觉趣味",
        },
        {
            "when_high": "stay",
            "adjust": {"green_view": +0.06, "edge_density": -0.02, "skyline_variance": +0.01},
            "layout_hint": "强化可坐停留节点与边界绿化，形成停留锚点",
        },
        {
            "when_low": "safety",
            "adjust": {"green_view": -0.05, "sky_view": +0.06},
            "layout_hint": "避免过密植被造成视线遮挡与荫蔽压迫",
        },
    ],
}


class KnowledgeBase:
    """本地 JSON 知识库：读写记忆 + 余弦相似度检索。"""

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = Path(data_dir or KB_DIR)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.memories_path = self.data_dir / "memories.json"
        self.rules_path = self.data_dir / "mapping_rules.json"
        self._ensure_files()

    def _ensure_files(self) -> None:
        if not self.memories_path.exists():
            self._write_json(self.memories_path, {"records": []})
        if not self.rules_path.exists():
            self._write_json(self.rules_path, DEFAULT_MAPPING_RULES)

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _read_json(path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    # ---------- 规则 ----------
    def get_mapping_rules(self) -> dict[str, Any]:
        return self._read_json(self.rules_path)

    # ---------- 记忆 CRUD ----------
    def list_memories(self) -> list[dict[str, Any]]:
        return self._read_json(self.memories_path).get("records", [])

    def add_memory(self, record: MemoryRecord | dict[str, Any]) -> MemoryRecord:
        if isinstance(record, dict):
            if "id" not in record or not record["id"]:
                record["id"] = str(uuid.uuid4())
            mem = MemoryRecord(**record)
        else:
            mem = record
            if not mem.id:
                mem.id = str(uuid.uuid4())

        data = self._read_json(self.memories_path)
        payload = mem.model_dump()
        payload["created_at"] = datetime.now(timezone.utc).isoformat()
        data.setdefault("records", []).append(payload)
        self._write_json(self.memories_path, data)
        return mem

    def update_memory(self, memory_id: str, **fields: Any) -> Optional[dict[str, Any]]:
        data = self._read_json(self.memories_path)
        for rec in data.get("records", []):
            if rec.get("id") == memory_id:
                rec.update(fields)
                self._write_json(self.memories_path, data)
                return rec
        return None

    # ---------- Few-shot 检索 ----------
    @staticmethod
    def _knob_vector(knobs: dict[str, float]) -> list[float]:
        return [float(knobs.get(k, 3.0)) for k in EXPERIENCE_KEYS]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1e-9
        nb = math.sqrt(sum(x * x for x in b)) or 1e-9
        return dot / (na * nb)

    def retrieve_similar(self, knobs: dict[str, float], top_k: int = 2) -> list[dict[str, Any]]:
        """用当前旋钮向量与历史 knobs 做余弦相似度，返回 Top-K。"""
        query = self._knob_vector(knobs)
        scored = []
        for rec in self.list_memories():
            vec = self._knob_vector(rec.get("knobs") or {})
            scored.append((self._cosine(query, vec), rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_k]]


# 默认单例，方便 Agent 直接 import
kb = KnowledgeBase()
