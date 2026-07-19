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
    RAG_EMBEDDING_API_KEY,
    RAG_EMBEDDING_BASE_URL,
    RAG_EMBEDDING_BATCH_SIZE,
    RAG_EMBEDDING_DIMENSIONS,
    RAG_EMBEDDING_FALLBACK_TO_TFIDF,
    RAG_EMBEDDING_MODEL,
    RAG_EMBEDDING_TIMEOUT,
    RAG_PUBLISHED_KNOWLEDGE_DIR,
    RAG_ENABLED,
    RAG_INCLUDE_REPOSITORY_SOURCES,
    RAG_MIN_SCORE,
    RAG_RETRIEVAL_MODE,
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

    def _load_chunks(self) -> list[dict[str, Any]]:
        """读取并切分知识源；不在此处决定使用 TF-IDF 还是 Embedding。"""
        if self._chunks:
            return self._chunks
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
        return self._chunks

    def _ensure_index(self) -> None:
        if not self.enabled or self._matrix is not None:
            return
        chunks = self._load_chunks()
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
            record_metadata = {"type": record_type, "record_id": record_id}
            if isinstance(record, dict):
                for key in ("case_type", "scene_type", "scene_label", "review_status"):
                    value = record.get(key)
                    if isinstance(value, (str, int, float, bool)):
                        record_metadata[key] = value
            chunks.append(
                self._chunk(
                    path,
                    f"{path.stem}:{record_id}",
                    text,
                    record_metadata,
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

    def _rank_results(
        self,
        base_scores: Any,
        *,
        is_task2: bool,
        kwargs: dict[str, Any],
        retrieval_mode: str,
    ) -> list[dict[str, Any]]:
        """保留 Task 2/3 的来源配额和场景案例约束，供两种检索器共用。"""
        import numpy as np

        if len(base_scores) != len(self._chunks):
            raise ValueError("RAG 相似度数量与知识块数量不一致")
        adjusted_scores = np.asarray(base_scores, dtype=float).copy()
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
                elif source_name == "scene_prompt_examples.json":
                    # 场景案例是 Task 3 的少样本参考；加权后仍在最终结果中限额，
                    # 避免 30 个案例挤掉通用空间规则。
                    adjusted_scores[index] += 0.18
                    scene_type = str(chunk.get("metadata", {}).get("scene_type") or "")
                    scene_context = str(kwargs.get("scene_context") or "").casefold()
                    scene_terms = {
                        "community": ("community", "社区", "居住", "住宅"),
                        "blue_green": ("blue_green", "蓝绿", "滨水", "水体", "公园"),
                        "commercial_office": (
                            "commercial_office",
                            "商办",
                            "办公",
                            "商业",
                        ),
                    }
                    if any(
                        term in scene_context for term in scene_terms.get(scene_type, ())
                    ):
                        adjusted_scores[index] += 0.08
                elif source_name in {"memories.json", "learning_feedback.json"}:
                    adjusted_scores[index] += 0.05
                elif source_name == "task2_3_framework.md":
                    adjusted_scores[index] += 0.03
        ranked = adjusted_scores.argsort()[::-1]
        eligible = [
            int(index)
            for index in ranked
            if float(adjusted_scores[index]) >= self.min_score
            and not (
                is_task2
                and self._chunks[int(index)].get("metadata", {}).get("case_type")
                == "task3_scene_prompt_example"
            )
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
        source_counts: dict[str, int] = {}
        for index in ordered:
            score = min(1.0, float(adjusted_scores[index]))
            chunk = self._chunks[int(index)]
            source_name = Path(chunk["source"]).name.lower()
            source_limit = (
                3
                if not is_task2 and source_name == "scene_prompt_examples.json"
                else None
            )
            if source_limit is not None and source_counts.get(source_name, 0) >= source_limit:
                continue
            metadata = dict(chunk["metadata"])
            metadata.update(
                {
                    "retrieval_task": "task2" if is_task2 else "task3",
                    "base_similarity": round(float(base_scores[index]), 6),
                    "retrieval_mode": retrieval_mode,
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
            source_counts[source_name] = source_counts.get(source_name, 0) + 1
            if len(results) >= self.top_k:
                break
        return results

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
        return self._rank_results(
            base_scores,
            is_task2=is_task2,
            kwargs=kwargs,
            retrieval_mode="tfidf",
        )


class QwenEmbeddingRagProvider(LocalTfidfRagProvider):
    """使用 DashScope OpenAI-compatible Embedding API 的语义检索器。

    文档向量按知识块内容、模型和维度计算指纹并缓存到项目输出目录；缓存不含
    API Key。查询向量只在当前进程内缓存，避免会话内重复调用。发生配置或网络
    异常时可回退到父类 TF-IDF 检索，使前端流程不中断。
    """

    retrieval_mode = "qwen_embedding"

    def __init__(
        self,
        source_paths: list[str | Path] | None = None,
        *,
        api_key: str = RAG_EMBEDDING_API_KEY,
        base_url: str = RAG_EMBEDDING_BASE_URL,
        model: str = RAG_EMBEDDING_MODEL,
        dimensions: int = RAG_EMBEDDING_DIMENSIONS,
        batch_size: int = RAG_EMBEDDING_BATCH_SIZE,
        timeout: float = RAG_EMBEDDING_TIMEOUT,
        fallback_to_tfidf: bool = RAG_EMBEDDING_FALLBACK_TO_TFIDF,
        embedding_client: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(source_paths, **kwargs)
        self.api_key = str(api_key or "").strip()
        self.base_url = str(base_url or "").rstrip("/")
        self.model = str(model or "text-embedding-v4").strip()
        self.dimensions = max(64, int(dimensions))
        self.batch_size = max(1, int(batch_size))
        self.timeout = max(1.0, float(timeout))
        self.fallback_to_tfidf = bool(fallback_to_tfidf)
        self._embedding_client = embedding_client
        self._embedding_matrix: Any = None
        self._query_vectors: dict[str, Any] = {}
        self.last_embedding_error = ""

    @property
    def embedding_cache_dir(self) -> Path:
        return self.cache_dir / "qwen_embeddings"

    def _embedding_cache_path(self) -> Path:
        payload = {
            "version": 1,
            "model": self.model,
            "dimensions": self.dimensions,
            "chunks": [
                {
                    "chunk_id": item["chunk_id"],
                    "text": item["text"],
                    "source": item["source"],
                }
                for item in self._chunks
            ],
        }
        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return self.embedding_cache_dir / f"{digest}.json"

    @staticmethod
    def _normalise_vector(values: Any, *, dimensions: int) -> Any:
        import numpy as np

        vector = np.asarray(values, dtype=float)
        if vector.ndim != 1 or vector.size != dimensions:
            raise ValueError(
                f"Embedding 维度异常：期望 {dimensions}，实际 {vector.size}"
            )
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            raise ValueError("Embedding 向量不能为零向量")
        return vector / norm

    def _get_client(self) -> Any:
        if self._embedding_client is not None:
            return self._embedding_client
        if not self.api_key:
            raise RuntimeError("未配置 RAG_EMBEDDING_API_KEY 或 LLM_API_KEY")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("缺少 openai 依赖，无法调用 Qwen Embedding") from exc
        self._embedding_client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        return self._embedding_client

    def _embed_texts(self, texts: list[str]) -> list[Any]:
        if not texts:
            return []
        client = self._get_client()
        vectors: list[Any] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=self.dimensions,
                encoding_format="float",
            )
            data = sorted(response.data, key=lambda item: int(item.index))
            if len(data) != len(batch):
                raise RuntimeError("Qwen Embedding 返回条目数量不匹配")
            vectors.extend(
                self._normalise_vector(item.embedding, dimensions=self.dimensions)
                for item in data
            )
        return vectors

    def _load_cached_embeddings(self, cache_path: Path) -> Any | None:
        if not cache_path.is_file():
            return None
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                cached.get("model") != self.model
                or int(cached.get("dimensions", 0)) != self.dimensions
            ):
                return None
            vectors = cached.get("vectors")
            if not isinstance(vectors, list) or len(vectors) != len(self._chunks):
                return None
            import numpy as np

            return np.vstack(
                [
                    self._normalise_vector(vector, dimensions=self.dimensions)
                    for vector in vectors
                ]
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def _ensure_embedding_index(self) -> None:
        if not self.enabled or self._embedding_matrix is not None:
            return
        chunks = self._load_chunks()
        if not chunks:
            return
        cache_path = self._embedding_cache_path()
        cached = self._load_cached_embeddings(cache_path)
        if cached is not None:
            self._embedding_matrix = cached
            return
        vectors = self._embed_texts([item["text"] for item in chunks])
        if len(vectors) != len(chunks):
            raise RuntimeError("Qwen Embedding 文档向量数量不匹配")
        import numpy as np

        self._embedding_matrix = np.vstack(vectors)
        self.embedding_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "model": self.model,
                    "dimensions": self.dimensions,
                    "vectors": self._embedding_matrix.tolist(),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def _embedding_query(self, query: str) -> Any:
        cached = self._query_vectors.get(query)
        if cached is not None:
            return cached
        vectors = self._embed_texts([query])
        if len(vectors) != 1:
            raise RuntimeError("Qwen Embedding 查询向量数量不匹配")
        self._query_vectors[query] = vectors[0]
        return vectors[0]

    def _tfidf_fallback(self, **kwargs: Any) -> list[dict[str, Any]]:
        LocalTfidfRagProvider._ensure_index(self)
        if self._matrix is None or self._vectorizer is None or not self._chunks:
            return []
        is_task2 = "experience_records" in kwargs
        query = self._task2_query(kwargs) if is_task2 else self._task3_query(kwargs)
        from sklearn.metrics.pairwise import linear_kernel

        scores = linear_kernel(self._vectorizer.transform([query]), self._matrix).ravel()
        return self._rank_results(
            scores,
            is_task2=is_task2,
            kwargs=kwargs,
            retrieval_mode="tfidf_fallback",
        )

    def retrieve(self, **kwargs: Any) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        try:
            self._ensure_embedding_index()
            if self._embedding_matrix is None or not self._chunks:
                return []
            is_task2 = "experience_records" in kwargs
            query = self._task2_query(kwargs) if is_task2 else self._task3_query(kwargs)
            query_vector = self._embedding_query(query)
            scores = self._embedding_matrix @ query_vector
            return self._rank_results(
                scores,
                is_task2=is_task2,
                kwargs=kwargs,
                retrieval_mode=self.retrieval_mode,
            )
        except Exception as exc:
            self.last_embedding_error = str(exc)
            if self.fallback_to_tfidf:
                return self._tfidf_fallback(**kwargs)
            return []


def build_default_rag_provider(
) -> NullRagProvider | LocalTfidfRagProvider | QwenEmbeddingRagProvider:
    if not RAG_ENABLED:
        return NullRagProvider()
    if RAG_RETRIEVAL_MODE == "tfidf":
        return LocalTfidfRagProvider()
    if RAG_RETRIEVAL_MODE == "qwen_embedding" or RAG_EMBEDDING_API_KEY:
        return QwenEmbeddingRagProvider()
    return LocalTfidfRagProvider()


__all__ = [
    "LayoutRagProvider",
    "LocalTfidfRagProvider",
    "NullRagProvider",
    "QwenEmbeddingRagProvider",
    "TranslationRagProvider",
    "build_default_rag_provider",
]
