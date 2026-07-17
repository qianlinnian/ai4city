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

from config import (
    EXPERIENCE_DIRECTIONS,
    EXPERIENCE_KEYS,
    EXPERIENCE_SCALE,
    KB_DIR,
    normalize_experience_values,
)
from schemas.models import MemoryRecord


DEFAULT_MAPPING_RULES = {
    "version": "task2-3-v1",
    "description": "七项VR体感变化 → 七项形态要素目标的启发式初始参数",
    "scale": EXPERIENCE_SCALE,
    "disclaimer": "用于基础框架和演示，系启发式初值，后续应由专家样本与VR实验校准。",
    "rules": [
        {
            "experience_key": "comfort",
            "direction": "higher_is_better",
            "adjust_per_point": {
                "green_view": 0.030,
                "sky_view": 0.015,
                "built_ratio": -0.025,
                "edge_density": -0.008,
                "color_richness": 0.200
            },
            "layout_hint": "增加遮阴、柔和材质与适量绿化，降低硬质界面压迫",
        },
        {
            "experience_key": "naturalness",
            "direction": "higher_is_better",
            "adjust_per_point": {
                "green_view": 0.050,
                "blue_view": 0.010,
                "built_ratio": -0.020,
                "edge_density": -0.005,
                "color_richness": 0.300
            },
            "layout_hint": "采用乔灌草复层绿化和自然材质，增强亲自然特征",
        },
        {
            "experience_key": "safety",
            "direction": "higher_is_better",
            "adjust_per_point": {
                "sky_view": 0.030,
                "built_ratio": -0.015,
                "edge_density": -0.015,
                "skyline_variance": -0.002
            },
            "layout_hint": "打开视线通廊，减少死角与过密遮挡，提升界面整洁",
        },
        {
            "experience_key": "relaxation",
            "direction": "higher_is_better",
            "adjust_per_point": {
                "green_view": 0.040,
                "blue_view": 0.010,
                "edge_density": -0.012,
                "color_richness": -0.100
            },
            "layout_hint": "弱化视觉噪声，增加连续绿化与安静的半围合停留空间",
        },
        {
            "experience_key": "environmental_disturbance",
            "direction": "lower_is_better",
            "adjust_per_point": {
                "green_view": 0.020,
                "built_ratio": -0.020,
                "edge_density": -0.012,
                "color_richness": -0.200
            },
            "layout_hint": "以绿化缓冲和界面整合减少交通、标识与杂乱设施干扰",
        },
        {
            "experience_key": "stay_intention",
            "direction": "higher_is_better",
            "adjust_per_point": {
                "green_view": 0.025,
                "sky_view": -0.010,
                "built_ratio": -0.010,
                "color_richness": 0.150,
                "skyline_variance": 0.003
            },
            "layout_hint": "强化可坐停留节点与边界绿化，形成停留锚点",
        },
        {
            "experience_key": "overall_impression",
            "direction": "higher_is_better",
            "adjust_per_point": {
                "green_view": 0.020,
                "blue_view": 0.005,
                "sky_view": 0.005,
                "built_ratio": -0.010,
                "edge_density": -0.005,
                "color_richness": 0.200
            },
            "layout_hint": "综合协调绿化、水体、开敞度与材质色彩，保持整体一致性",
        },
    ],
}


DEFAULT_EXPERIENCE_CASES = {
    "version": "task2-3-v1",
    "description": "由仓库 p1/p2 图像特征与历史启发式评分整理的基础 RAG 示例，非实测结论。",
    "records": [],
}


class KnowledgeBase:
    """本地 JSON 知识库：读写记忆 + 余弦相似度检索。"""

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = Path(data_dir or KB_DIR)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.memories_path = self.data_dir / "memories.json"
        self.rules_path = self.data_dir / "mapping_rules.json"
        self.experience_cases_path = self.data_dir / "experience_morph_cases.json"
        self._ensure_files()

    def _ensure_files(self) -> None:
        if not self.memories_path.exists():
            self._write_json(self.memories_path, {"records": []})
        if not self.rules_path.exists():
            self._write_json(self.rules_path, DEFAULT_MAPPING_RULES)
        if not self.experience_cases_path.exists():
            self._write_json(self.experience_cases_path, DEFAULT_EXPERIENCE_CASES)

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _read_json(path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    # ---------- 规则 ----------
    def get_mapping_rules(self) -> dict[str, Any]:
        return self._read_json(self.rules_path)

    def list_experience_cases(self) -> list[dict[str, Any]]:
        """返回随代码维护的基础案例，不与运行时反馈混写。"""
        return self._read_json(self.experience_cases_path).get("records", [])

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
        normalized = normalize_experience_values(knobs)
        return [normalized[k] for k in EXPERIENCE_KEYS]

    @staticmethod
    def _delta_vector(
        experience_baseline: dict[str, float],
        experience_targets: dict[str, float],
    ) -> list[float]:
        baseline = normalize_experience_values(experience_baseline)
        targets = normalize_experience_values(experience_targets)
        vector = []
        for key in EXPERIENCE_KEYS:
            delta = targets[key] - baseline[key]
            if EXPERIENCE_DIRECTIONS[key] == "lower_is_better":
                delta = -delta
            vector.append(delta)
        return vector

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1e-9
        nb = math.sqrt(sum(x * x for x in b)) or 1e-9
        return dot / (na * nb)

    def retrieve_experience_cases(
        self,
        experience_baseline: dict[str, float],
        experience_targets: dict[str, float],
        scene_context: str = "",
        top_k: int = 2,
    ) -> list[dict[str, Any]]:
        """按七项体感“改善方向向量”检索基础案例与历史人工反馈。"""
        query = self._delta_vector(experience_baseline, experience_targets)
        candidates = list(self.list_experience_cases())
        candidates.extend(self.list_memories())
        scene_chars = set(scene_context.replace(" ", ""))
        scored: list[tuple[float, dict[str, Any]]] = []

        for rec in candidates:
            rec_baseline = rec.get("experience_baseline") or {
                key: EXPERIENCE_SCALE["neutral"] for key in EXPERIENCE_KEYS
            }
            rec_targets = rec.get("experience_targets") or rec.get("knobs") or {}
            vector = self._delta_vector(rec_baseline, rec_targets)
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(query, vector)))
            max_distance = 4.0 * math.sqrt(len(EXPERIENCE_KEYS))
            score = max(0.0, 1.0 - distance / max_distance)

            rec_scene = str(rec.get("scene_context") or "").replace(" ", "")
            if scene_chars and rec_scene:
                overlap = len(scene_chars & set(rec_scene)) / max(1, len(scene_chars))
                score = min(1.0, score + 0.10 * overlap)

            enriched = dict(rec)
            enriched["_rag_score"] = round(score, 4)
            scored.append((score, enriched))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[:top_k]]

    def retrieve_similar(self, knobs: dict[str, float], top_k: int = 2) -> list[dict[str, Any]]:
        """兼容旧接口：以中性 3 分为体感基线进行检索。"""
        neutral = {key: EXPERIENCE_SCALE["neutral"] for key in EXPERIENCE_KEYS}
        return self.retrieve_experience_cases(neutral, knobs, top_k=top_k)


# 默认单例，方便 Agent 直接 import
kb = KnowledgeBase()
