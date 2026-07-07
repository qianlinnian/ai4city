from __future__ import annotations

import tempfile
from pathlib import Path


def default_data_root() -> Path:
    root = Path(tempfile.gettempdir()) / "ai4city_mas"
    root.mkdir(parents=True, exist_ok=True)
    return root


def default_chroma_dir() -> str:
    path = default_data_root() / "chroma"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def default_memory_db() -> str:
    return str(default_data_root() / "memory_with_rag.db")


def default_hf_home() -> str:
    path = default_data_root() / "huggingface"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
