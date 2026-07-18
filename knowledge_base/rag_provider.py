"""Task 2/3 可插拔 RAG：默认关闭，本地 TF-IDF 可选开启。"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from config import (
    DATA_DIR,
    KNOWLEDGE_SOURCE_DIR,
    RAG_CACHE_DIR,
    RAG_PUBLISHED_KNOWLEDGE_DIR,
    RAG_ENABLED,
    RAG_INCLUDE_REPOSITORY_SOURCES,
    RAG_MIN_SCORE,
    RAG_TOP_K,
    ROOT,
)
from schemas.models import RagSearchResult


@runtime_checkable
class TranslationRagProvider(Protocol):
    enabled: bool

    def retrieve(
        self,
        *,
        experience_records: list[dict[str, Any]],
        experience_targets: dict[str, float],
        baseline_metrics: dict[str, float],
        scene_context: str,
    ) -> list[dict[str, Any]]: ...


@runtime_checkable
class LayoutRagProvider(Protocol):
    enabled: bool

    def retrieve(
        self,
        *,
        baseline_metrics: dict[str, float],
        target_metrics: dict[str, float],
        scene_context: str,
        expert_advice: str,
        scene_understanding: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...


class NullRagProvider:
    """显式关闭的 Provider；不读取知识文件，也不执行检索。"""

    enabled = False

    def retrieve(self, **_: Any) -> list[dict[str, Any]]:
        return []


class LocalTfidfRagProvider:
    """无需 Embedding API 的本地字符 TF-IDF 检索器。"""

    enabled = True

    def __init__(
        self,
        source_paths: list[str | Path] | None = None,
        *,
        top_k: int = RAG_TOP_K,
        min_score: float = RAG_MIN_SCORE,
        chunk_size: int = 900,
        enabled: bool = True,
        cache_dir: str | Path = RAG_CACHE_DIR,
    ) -> None:
        self.enabled = bool(enabled)
        self.top_k = max(1, int(top_k))
        self.min_score = max(0.0, float(min_score))
        self.chunk_size = max(200, int(chunk_size))
        self.source_paths = [Path(item).resolve() for item in (
            source_paths if source_paths is not None else self.default_source_paths()
        )]
        self.cache_dir = Path(cache_dir).resolve()
        if self.cache_dir == DATA_DIR or self.cache_dir.is_relative_to(DATA_DIR):
            raise ValueError("RAG 派生缓存目录不得位于只读数据目录中")
        self._chunks: list[dict[str, Any]] = []
        self._vectorizer: Any = None
        self._matrix: Any = None

    @staticmethod
    def default_source_paths() -> list[Path]:
        candidates: list[Path] = []
        if RAG_PUBLISHED_KNOWLEDGE_DIR.is_dir():
            candidates.extend(sorted(RAG_PUBLISHED_KNOWLEDGE_DIR.rglob("*.json")))
        if KNOWLEDGE_SOURCE_DIR.is_dir():
            candidates.extend(sorted(KNOWLEDGE_SOURCE_DIR.rglob("*.pdf")))
            candidates.extend(sorted(KNOWLEDGE_SOURCE_DIR.rglob("*.txt")))
            candidates.extend(sorted(KNOWLEDGE_SOURCE_DIR.rglob("*.md")))
        if RAG_INCLUDE_REPOSITORY_SOURCES:
            data_dir = ROOT / "knowledge_base" / "data"
            if data_dir.is_dir():
                candidates.extend(
                    path for path in sorted(data_dir.iterdir())
                    if path.suffix.lower() in {".json", ".md", ".txt"}
                )
            candidates.extend(
                [
                    ROOT / "docs" / "TASK2_3_FRAMEWORK.md",
                    ROOT / "_extracted_text" / "metrics_definition.txt",
                ]
            )
        return [path for path in candidates if path.is_file()]

    @property
    def indexed_chunk_count(self) -> int:
        return len(self._chunks)

    def _ensure_index(self) -> None:
        if not self.enabled or self._matrix is not None:
            return
        chunks: list[dict[str, Any]] = []
        for path in self.source_paths:
            if not path.is_file():
                continue
            try:
                if path.suffix.lower() == ".json":
                    chunks.extend(self._json_chunks(path))
                elif path.suffix.lower() == ".pdf":
                    chunks.extend(self._pdf_chunks(path))
                else:
                    chunks.extend(self._text_chunks(path))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        self._chunks = chunks
        if not chunks:
            return
        from sklearn.feature_extraction.text import TfidfVectorizer

        # 字符 n-gram 对中英文混合规则、文件名与指标键都较稳健，无需分词模型。
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            lowercase=True,
            sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform(
            [item["text"] for item in chunks]
        )

    def _json_chunks(self, path: Path) -> list[dict[str, Any]]:
        data = json.loads(path.read_text(encoding="utf-8"))
        records: list[Any]
        record_type = "document"
        if isinstance(data, dict) and isinstance(data.get("rules"), list):
            records = data["rules"]
            record_type = "rule"
        elif isinstance(data, dict) and isinstance(data.get("records"), list):
            records = data["records"]
            record_type = "case"
        else:
            records = [data]
        chunks = []
        for index, record in enumerate(records):
            text = json.dumps(record, ensure_ascii=False, sort_keys=True)
            if not text.strip() or text.strip() in {"{}", "[]"}:
                continue
            record_id = (
                str(record.get("id"))
                if isinstance(record, dict) and record.get("id")
                else f"{record_type}-{index + 1}"
            )
            chunks.append(
                self._chunk(
                    path,
                    f"{path.stem}:{record_id}",
                    text,
                    {"type": record_type, "record_id": record_id},
                )
            )
        return chunks

    def _text_chunks(self, path: Path) -> list[dict[str, Any]]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        paragraphs = [item.strip() for item in text.splitlines() if item.strip()]
        groups: list[str] = []
        current = ""
        for paragraph in paragraphs:
            if current and len(current) + len(paragraph) + 1 > self.chunk_size:
                groups.append(current)
                current = ""
            current = f"{current}\n{paragraph}".strip()
        if current:
            groups.append(current)
        return [
            self._chunk(
                path,
                f"{path.stem}:text-{index + 1}",
                group,
                {"type": "document", "part": index + 1},
            )
            for index, group in enumerate(groups)
        ]

    def _pdf_chunks(self, path: Path) -> list[dict[str, Any]]:
        """只读提取 PDF，并按真实页码切块；结果缓存到项目输出目录。"""
        stat = path.stat()
        cache_key = json.dumps(
            {
                "path": str(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "chunk_size": self.chunk_size,
                "extractor": "pdf-pages-v2",
            },
            sort_keys=True,
        )
        cache_name = hashlib.sha256(cache_key.encode("utf-8")).hexdigest() + ".json"
        cache_path = self.cache_dir / cache_name
        if cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("cache_key") == cache_key:
                    return list(cached.get("chunks") or [])
            except (OSError, ValueError, json.JSONDecodeError):
                pass

        pages = self._extract_pdf_pages(path)
        chunks: list[dict[str, Any]] = []
        for page_number, page_text in enumerate(pages, start=1):
            clean = "\n".join(
                line.rstrip() for line in page_text.splitlines() if line.strip()
            ).strip()
            if len(clean) < 30:
                continue
            parts = [
                clean[offset : offset + self.chunk_size]
                for offset in range(0, len(clean), self.chunk_size)
            ]
            for part_number, part in enumerate(parts, start=1):
                chunks.append(
                    self._chunk(
                        path,
                        f"{path.stem}:page-{page_number:04d}-part-{part_number:02d}",
                        part,
                        {
                            "type": "document",
                            "format": "pdf",
                            "page": page_number,
                            "part": part_number,
                            "source_group": "external_knowledge",
                        },
                    )
                )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "cache_key": cache_key,
                    "source": str(path),
                    "page_count": len(pages),
                    "searchable_page_count": len(
                        {item["metadata"]["page"] for item in chunks}
                    ),
                    "chunks": chunks,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return chunks

    @staticmethod
    def _extract_pdf_pages(path: Path) -> list[str]:
        executable = shutil.which("pdftotext")
        extractor_detail = ""
        if executable:
            try:
                creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                completed = subprocess.run(
                    [executable, "-layout", "-enc", "UTF-8", str(path), "-"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=300,
                    creationflags=creation_flags,
                )
                if completed.returncode == 0:
                    text = completed.stdout.decode("utf-8", errors="replace")
                    pages = text.split("\f")
                    if pages and not pages[-1].strip():
                        pages.pop()
                    return pages
                extractor_detail = completed.stderr.decode(
                    "utf-8", errors="replace"
                )
            except (OSError, subprocess.SubprocessError) as exc:
                extractor_detail = str(exc)
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            return [page.extract_text() or "" for page in reader.pages]
        except Exception as exc:
            raise RuntimeError(
                f"PDF 文本提取失败: {path.name}; {extractor_detail or exc}"
            ) from exc

    def _chunk(
        self,
        path: Path,
        chunk_id: str,
        text: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            source = str(path.relative_to(ROOT))
        except ValueError:
            source = str(path)
        enriched_metadata = dict(metadata)
        if path == RAG_PUBLISHED_KNOWLEDGE_DIR or path.is_relative_to(
            RAG_PUBLISHED_KNOWLEDGE_DIR
        ):
            enriched_metadata["source_group"] = "curated_knowledge"
        elif path == KNOWLEDGE_SOURCE_DIR or path.is_relative_to(KNOWLEDGE_SOURCE_DIR):
            enriched_metadata["source_group"] = "external_knowledge"
        else:
            enriched_metadata.setdefault("source_group", "repository_supplement")
        return {
            "text": text,
            "source": source,
            "chunk_id": chunk_id,
            "metadata": enriched_metadata,
        }

    @staticmethod
    def _task2_query(kwargs: dict[str, Any]) -> str:
        return " ".join(
            [
                "Task2 七项体感目标到七项形态目标 指标变化合理性",
                "experience morphology comfort nature safety relaxation disturbance stay green blue sky built color edge skyline",
                json.dumps(kwargs.get("experience_records") or [], ensure_ascii=False),
                json.dumps(kwargs.get("experience_targets") or {}, ensure_ascii=False),
                json.dumps(kwargs.get("baseline_metrics") or {}, ensure_ascii=False),
                str(kwargs.get("scene_context") or ""),
            ]
        )

    @staticmethod
    def _task3_query(kwargs: dict[str, Any]) -> str:
        return " ".join(
            [
                "Task3 空间对象布置 位置 数量 空间关系 保持区域 全景接缝",
                "spatial layout object position quantity relation fixed region panorama seam park entrance pedestrian road fire access accessibility vegetation building infrastructure",
                json.dumps(kwargs.get("baseline_metrics") or {}, ensure_ascii=False),
                json.dumps(kwargs.get("target_metrics") or {}, ensure_ascii=False),
                str(kwargs.get("scene_context") or ""),
                str(kwargs.get("expert_advice") or ""),
                json.dumps(kwargs.get("scene_understanding") or {}, ensure_ascii=False),
            ]
        )

    def retrieve(self, **kwargs: Any) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        self._ensure_index()
        if self._matrix is None or self._vectorizer is None or not self._chunks:
            return []
        is_task2 = "experience_records" in kwargs
        query = (
            self._task2_query(kwargs)
            if is_task2
            else self._task3_query(kwargs)
        )
        from sklearn.metrics.pairwise import linear_kernel

        query_vector = self._vectorizer.transform([query])
        base_scores = linear_kernel(query_vector, self._matrix).ravel()
        adjusted_scores = base_scores.copy()
        for index, chunk in enumerate(self._chunks):
            source_name = Path(chunk["source"]).name.lower()
            base_score = float(base_scores[index])
            source_group = chunk.get("metadata", {}).get("source_group")
            if source_group == "curated_knowledge" and base_score > 0.01:
                adjusted_scores[index] += 0.18
            elif source_group == "external_knowledge" and base_score > 0.01:
                adjusted_scores[index] += 0.12
            if is_task2:
                if source_name == "mapping_rules.json":
                    adjusted_scores[index] += 0.08
                elif "metrics_definition" in source_name or source_name == "task2_3_framework.md":
                    adjusted_scores[index] += 0.06
                elif source_name == "experience_morph_cases.json":
                    adjusted_scores[index] += 0.03
            else:
                if source_name == "spatial_rules.json":
                    adjusted_scores[index] += 0.15
                elif source_name in {"memories.json", "learning_feedback.json"}:
                    adjusted_scores[index] += 0.05
                elif source_name == "task2_3_framework.md":
                    adjusted_scores[index] += 0.03
        ranked = adjusted_scores.argsort()[::-1]
        eligible = [
            int(index)
            for index in ranked
            if float(adjusted_scores[index]) >= self.min_score
        ]
        external = [
            index
            for index in eligible
            if self._chunks[index].get("metadata", {}).get("source_group")
            in {"curated_knowledge", "external_knowledge"}
        ]
        supplements = [index for index in eligible if index not in set(external)]
        external_quota = min(len(external), max(1, (self.top_k * 3 + 3) // 4))
        ordered = [
            *external[:external_quota],
            *supplements,
            *external[external_quota:],
        ]
        results: list[dict[str, Any]] = []
        for index in ordered:
            score = min(1.0, float(adjusted_scores[index]))
            chunk = self._chunks[int(index)]
            metadata = dict(chunk["metadata"])
            metadata.update(
                {
                    "retrieval_task": "task2" if is_task2 else "task3",
                    "base_similarity": round(float(base_scores[index]), 6),
                }
            )
            result = RagSearchResult(
                text=chunk["text"],
                source=chunk["source"],
                chunk_id=chunk["chunk_id"],
                score=round(score, 6),
                metadata=metadata,
            ).model_dump()
            # 兼容现有 Agent 的引用字段，同时保留正式 chunk_id。
            result["id"] = result["chunk_id"]
            results.append(result)
            if len(results) >= self.top_k:
                break
        return results


def build_default_rag_provider() -> NullRagProvider | LocalTfidfRagProvider:
    return LocalTfidfRagProvider() if RAG_ENABLED else NullRagProvider()


__all__ = [
    "LayoutRagProvider",
    "LocalTfidfRagProvider",
    "NullRagProvider",
    "TranslationRagProvider",
    "build_default_rag_provider",
]
