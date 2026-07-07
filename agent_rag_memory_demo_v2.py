from __future__ import annotations

import argparse
import os
import re
import warnings
from pathlib import Path
from typing import Iterable

import httpx
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import Annotated, TypedDict

from app_paths import default_chroma_dir, default_hf_home, default_memory_db
from local_embeddings import LocalHashEmbeddings

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    HuggingFaceEmbeddings = None


warnings.filterwarnings(
    "ignore",
    message="The default value of `allowed_objects` will change in a future version.*",
)

load_dotenv()

HF_HOME = os.getenv("HF_HOME") or default_hf_home()
os.environ["HF_HOME"] = HF_HOME
os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(HF_HOME) / "transformers"))
os.environ["NO_PROXY"] = ",".join(
    filter(None, [os.getenv("NO_PROXY", ""), "127.0.0.1", "localhost"])
)
for proxy_key in (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
):
    os.environ.pop(proxy_key, None)

LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "local-model")
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "local_hash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
CHROMA_DIR = os.getenv("CHROMA_DIR") or default_chroma_dir()
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "ai4city_docs")
MEMORY_DB = os.getenv("MEMORY_DB") or default_memory_db()
DEFAULT_THREAD_ID = os.getenv("THREAD_ID", "boss_local_session_001")
DEFAULT_TOP_K = int(os.getenv("TOP_K", "4"))

METRIC_PATTERN = re.compile(
    r"("
    r"\d+(?:\.\d+)?%\s*(?:~|～|-|至)\s*\d+(?:\.\d+)?%|"
    r"GVF\s*=\s*\d+(?:\.\d+)?%|"
    r"\d+(?:\.\d+)?%|"
    r"β\s*=\s*\d+(?:\.\d+)?|"
    r"p\s*[<>＝=]\s*0?\.\d+|"
    r"RMSSD|HRV|EDA"
    r")",
    re.IGNORECASE,
)

METRIC_HINTS = (
    "指标",
    "阈值",
    "区间",
    "GVF",
    "HRV",
    "RMSSD",
    "EDA",
    "beta",
    "β",
    "作用域",
    "边际递减",
    "效益",
    "显著",
)

print("正在初始化系统组件，请稍候...")

llm = ChatOpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key=LM_STUDIO_API_KEY,
    model=LM_STUDIO_MODEL,
    temperature=0.1,
)

if EMBEDDING_BACKEND == "huggingface":
    if HuggingFaceEmbeddings is None:
        raise RuntimeError("当前环境不可用 HuggingFaceEmbeddings。")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
else:
    embeddings = LocalHashEmbeddings()

vector_store = Chroma(
    persist_directory=CHROMA_DIR,
    collection_name=CHROMA_COLLECTION,
    embedding_function=embeddings,
    client_settings=Settings(
        anonymized_telemetry=False,
        is_persistent=True,
        persist_directory=CHROMA_DIR,
    ),
)

RUNTIME_TOP_K = DEFAULT_TOP_K


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local RAG + memory demo")
    parser.add_argument("--thread-id", default=DEFAULT_THREAD_ID)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--reset-memory",
        action="store_true",
        help="Delete the SQLite memory DB before starting the session.",
    )
    return parser.parse_args()


def split_sentences(text: str) -> Iterable[str]:
    for item in re.split(r"(?<=[。！？；;])\s*", text):
        sentence = item.strip()
        if sentence:
            yield sentence


def extract_metric_lines(docs) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for doc in docs:
        source = doc.metadata.get("file_name", "unknown")
        chunk_index = doc.metadata.get("chunk_index", "?")
        for sentence in split_sentences(doc.page_content):
            normalized = re.sub(r"\s+", " ", sentence)
            if not normalized:
                continue
            has_metric = bool(METRIC_PATTERN.search(normalized))
            has_hint = any(hint.lower() in normalized.lower() for hint in METRIC_HINTS)
            if not (has_metric or has_hint):
                continue
            line = f"- {source} / chunk {chunk_index}: {normalized}"
            if line not in seen:
                seen.add(line)
                lines.append(line)
    return lines[:8]


@tool
def retrieve_local_knowledge(query: str) -> str:
    """
    当问题涉及本地文档中的预算、指标、阈值、项目事实、流程或技术细节时，调用该工具检索知识库。
    """
    print(f"\n[动作] AI 正在翻阅本地知识库，搜索关键词: {query}")
    docs = vector_store.similarity_search(query, k=RUNTIME_TOP_K)
    if not docs:
        return "本地知识库中未找到相关内容。"

    metric_lines = extract_metric_lines(docs)
    sections = []
    for doc in docs:
        source = doc.metadata.get("file_name", "unknown")
        chunk_index = doc.metadata.get("chunk_index", "?")
        sections.append(f"来源: {source} / chunk {chunk_index}\n{doc.page_content}")

    parts = []
    if metric_lines:
        parts.append("候选指标/阈值/数值:\n" + "\n".join(metric_lines))
    parts.append("检索原文片段:\n" + "\n\n".join(sections))
    return "\n\n".join(parts)


tools = [retrieve_local_knowledge]
llm_with_tools = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def assistant_node(state: AgentState) -> dict:
    sys_msg = SystemMessage(
        content=(
            "你是一个安全、专业的高级本地 AI 助手。\n"
            "1. 普通寒暄和身份记忆问题直接回答，不要调用检索工具。\n"
            "2. 当用户询问本地文档中的预算、指标、阈值、项目事实、工作营安排或技术细节时，必须调用检索工具。\n"
            "3. 如果检索结果里出现“候选指标/阈值/数值”，你必须优先从这些内容中抽取明确答案，尽量直接给出数值、区间、阈值和来源。\n"
            "4. 如果文档没有明确数值，就明确说“文档未直接给出明确数值”，不要编造。\n"
            "5. 回答文档问题时，优先使用“结论 + 证据来源”的格式，必要时列 1 到 3 个关键指标。\n"
            "6. 记住用户的名字、身份和最近讨论过的项目。\n"
            "7. 全程用中文作答。"
        )
    )
    response = llm_with_tools.invoke([sys_msg] + state["messages"])
    return {"messages": [response]}


def build_app():
    builder = StateGraph(AgentState)
    builder.add_node("assistant", assistant_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    return builder


def ensure_lm_studio_ready() -> None:
    models_url = f"{LM_STUDIO_BASE_URL.rstrip('/')}/models"
    try:
        response = httpx.get(models_url, timeout=5.0)
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            "无法连接到 LM Studio 本地接口。\n"
            f"请确认 LM Studio 已启动，并且 OpenAI 兼容服务监听在 {LM_STUDIO_BASE_URL}。\n"
            f"健康检查地址: {models_url}\n"
            f"原始错误: {exc}"
        ) from exc


def reset_memory_db(memory_db: str) -> None:
    path = Path(memory_db)
    if path.exists():
        path.unlink()
        print(f"已清空记忆库: {path}")


def main() -> None:
    global RUNTIME_TOP_K

    args = parse_args()
    RUNTIME_TOP_K = args.top_k

    if not os.path.isdir(CHROMA_DIR):
        raise FileNotFoundError(
            f"未找到知识库目录 {CHROMA_DIR}，请先运行 python build_knowledge_base.py"
        )

    if args.reset_memory:
        reset_memory_db(MEMORY_DB)

    ensure_lm_studio_ready()

    with SqliteSaver.from_conn_string(MEMORY_DB) as memory:
        app = build_app().compile(checkpointer=memory)
        config = {"configurable": {"thread_id": args.thread_id}}

        print("\n" + "=" * 50)
        print("【本地私有化 Agent Demo】已启动")
        print(f"聊天记忆: {MEMORY_DB}")
        print(f"知识库目录: {CHROMA_DIR}")
        print(f"知识库集合: {CHROMA_COLLECTION}")
        print(f"向量后端: {EMBEDDING_BACKEND}")
        print(f"会话 ID: {args.thread_id}")
        print("=" * 50 + "\n")

        while True:
            try:
                user_input = input("你的输入（输入 q 退出）: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in {"q", "quit", "exit"}:
                    print("正在安全退出。")
                    break

                events = app.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config,
                    stream_mode="values",
                )

                final_text = None
                for event in events:
                    last_msg = event["messages"][-1]
                    if last_msg.type == "ai" and last_msg.content:
                        final_text = last_msg.content

                if final_text:
                    print(f"助手: {final_text}\n")
            except KeyboardInterrupt:
                print("\n已中断，正在退出。")
                break


if __name__ == "__main__":
    main()
