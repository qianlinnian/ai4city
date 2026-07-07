from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

from dotenv import load_dotenv
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_chroma import Chroma

from app_paths import default_chroma_dir, default_hf_home
from local_embeddings import LocalHashEmbeddings

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    HuggingFaceEmbeddings = None


WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def extract_docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        xml_bytes = archive.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    paragraphs = []
    for para in root.findall(".//w:body/w:p", WORD_NS):
        texts = [node.text or "" for node in para.findall(".//w:t", WORD_NS)]
        merged = "".join(texts).strip()
        if merged:
            paragraphs.append(merged)
    return "\n".join(paragraphs)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> Iterable[str]:
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        if end == len(text):
            break
        start = max(end - overlap, start + 1)


def main() -> None:
    load_dotenv()

    hf_home = os.getenv("HF_HOME") or default_hf_home()
    os.environ["HF_HOME"] = hf_home
    os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(hf_home) / "transformers"))

    docs_glob = os.getenv("DOCS_GLOB", "*.docx")
    chroma_dir = os.getenv("CHROMA_DIR") or default_chroma_dir()
    collection_name = os.getenv("CHROMA_COLLECTION", "ai4city_docs")
    embedding_backend = os.getenv("EMBEDDING_BACKEND", "local_hash")
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")

    root = Path.cwd()
    docx_files = sorted(root.glob(docs_glob))
    if not docx_files:
        print(f"未找到匹配 {docs_glob} 的 docx 文件。")
        return

    print("正在初始化向量模型...")
    if embedding_backend == "huggingface":
        if HuggingFaceEmbeddings is None:
            raise RuntimeError("当前环境不可用 HuggingFaceEmbeddings。")
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
    else:
        embeddings = LocalHashEmbeddings()
    vector_store = Chroma(
        persist_directory=chroma_dir,
        collection_name=collection_name,
        embedding_function=embeddings,
        client_settings=Settings(
            anonymized_telemetry=False,
            is_persistent=True,
            persist_directory=chroma_dir,
        ),
    )
    try:
        vector_store.delete_collection()
        vector_store = Chroma(
            persist_directory=chroma_dir,
            collection_name=collection_name,
            embedding_function=embeddings,
            client_settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True,
                persist_directory=chroma_dir,
            ),
        )
    except Exception:
        pass

    documents: list[Document] = []
    for path in docx_files:
        print(f"正在读取 {path.name}")
        text = extract_docx_text(path)
        if not text.strip():
            print(f"跳过空文档: {path.name}")
            continue
        for index, chunk in enumerate(chunk_text(text), start=1):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": str(path),
                        "file_name": path.name,
                        "chunk_index": index,
                    },
                )
            )

    if not documents:
        print("未生成任何知识切片。")
        return

    print(f"开始写入 Chroma，共 {len(documents)} 个切片...")
    vector_store.add_documents(documents)
    print(
        f"知识库构建完成，输出目录: {chroma_dir}，集合: {collection_name}，向量后端: {embedding_backend}"
    )


if __name__ == "__main__":
    main()
