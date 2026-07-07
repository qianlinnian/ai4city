# Local MAS Agent Demo

This project implements a local-first MAS-style agent demo aligned with the requirements in [task.md](D:/course/ai4city/task.md).

## What it does

- Connects to a local LM Studio chat model
- Persists conversation memory in SQLite through LangGraph
- Builds a local Chroma knowledge base from `.docx` files in this workspace
- Exposes local retrieval as a tool
- Supports restart-and-remember demos with a stable `thread_id`
- Uses a zero-download local embedding backend by default, with optional HuggingFace upgrade

## Files

- `agent_rag_memory_demo.py`: interactive agent entry point
- `build_knowledge_base.py`: ingests local `.docx` files into Chroma
- `environment.yml`: recommended conda environment
- `.env.example`: runtime configuration template
- `local_embeddings.py`: built-in offline embedding backend
- `app_paths.py`: default persistent paths under the system temp directory

## Expected demo flow

1. Introduce yourself as `张伟`, the R&D director.
2. Ask the agent about the budget or indicator for a project in the latest documents.
3. Quit with `q`, restart the script, and ask whether it still remembers you and the previous topic.

## Setup

```powershell
conda env create -f environment.yml
conda activate ai4city-mas
copy .env.example .env
python build_knowledge_base.py
python agent_rag_memory_demo.py
```

## Notes

- LM Studio must expose an OpenAI-compatible API at `http://127.0.0.1:1234/v1`.
- The default embedding backend is `local_hash`, so the demo can run without downloading an embedding model.
- If you want better retrieval quality later, set `EMBEDDING_BACKEND=huggingface` and keep `EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5`.
- Knowledge base ingestion scans the current workspace root for `.docx` files.
- By default, SQLite memory and Chroma persistence are stored under the system temp directory so they can write reliably in this environment.
