"""
LangChain LLM 适配层。

Task 2/3 通过 ``ChatPromptTemplate | ChatOpenAI | StrOutputParser`` 调用
OpenAI 兼容模型；多模态请求使用 LangChain 消息对象。无 API Key、未安装
LangChain 或远端调用失败时返回 ``None``，由各 Agent 走规则/RAG兜底路径。
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, use_mock_llm


def _langchain_components():
    """延迟导入，保证无模型依赖时规则/RAG演示仍可启动。"""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    return ChatOpenAI, ChatPromptTemplate, StrOutputParser, HumanMessage, SystemMessage


def _build_model(temperature: float):
    ChatOpenAI, *_ = _langchain_components()
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=temperature,
        timeout=120,
        max_retries=2,
    )


def _content_as_text(content: Any) -> str:
    """兼容 LangChain 返回的纯文本和标准内容块。"""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in {
                "text",
                "output_text",
            }:
                parts.append(str(block.get("text", "")))
        return "\n".join(part for part in parts if part).strip()
    return str(content or "").strip()


def chat(system: str, user: str, temperature: float = 0.4) -> Optional[str]:
    """
    调用聊天补全。
    返回: 文本；MOCK 或失败时返回 None（调用方应使用本地规则兜底）
    """
    if use_mock_llm():
        return None

    try:
        _, ChatPromptTemplate, StrOutputParser, *_ = _langchain_components()
        prompt = ChatPromptTemplate.from_messages(
            [("system", "{system}"), ("human", "{user}")]
        )
        chain = prompt | _build_model(temperature) | StrOutputParser()
        text = chain.invoke({"system": system, "user": user})
        return _content_as_text(text) or None
    except Exception as e:
        print(f"[LangChain] LLM 调用失败，回退本地规则/RAG: {e}")
        return None


def _image_data_url(image_path: str | Path, max_long_edge: int = 2048) -> str:
    """压缩本地图后生成 OpenAI 兼容的 data URL，避免直接上传超大全景。"""
    path = Path(image_path)
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        longest = max(width, height)
        if longest > max_long_edge:
            scale = max_long_edge / longest
            image = image.resize(
                (max(1, round(width * scale)), max(1, round(height * scale))),
                Image.Resampling.LANCZOS,
            )
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=82, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def chat_with_image(
    system: str,
    user: str,
    image_path: str | Path,
    temperature: float = 0.4,
) -> Optional[str]:
    """调用 OpenAI 兼容的多模态聊天补全；失败时由 Agent 走规则兜底。"""
    if use_mock_llm():
        return None

    path = Path(image_path)
    if not path.exists():
        print(f"[LLM] 原图不存在，改用纯文本调用: {path}")
        return chat(system, user, temperature=temperature)

    try:
        *_, HumanMessage, SystemMessage = _langchain_components()
        image_url = _image_data_url(path)
        response = _build_model(temperature).invoke(
            [
                SystemMessage(content=system),
                HumanMessage(
                    content=[
                        {"type": "text", "text": user},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ]
                ),
            ]
        )
        return _content_as_text(response.content) or None
    except Exception as e:
        print(f"[LangChain] 多模态调用失败，回退本地规则/RAG: {e}")
        return None


def _view_payload(view: Any) -> dict[str, Any]:
    if hasattr(view, "model_dump"):
        return dict(view.model_dump())
    if isinstance(view, dict):
        return dict(view)
    return {
        "view_id": Path(view).stem,
        "output_path": str(view),
        "is_overview": False,
    }


def build_multi_image_messages(
    system: str,
    user: str,
    views: list[Any],
) -> list[Any]:
    """构造带视图方位说明的 LangChain 多图消息，便于本地单元测试。"""
    *_, HumanMessage, SystemMessage = _langchain_components()
    content: list[dict[str, Any]] = [{"type": "text", "text": user}]
    valid_count = 0
    for raw_view in views:
        view = _view_payload(raw_view)
        path = Path(str(view.get("output_path") or ""))
        if not path.is_file():
            continue
        view_id = str(view.get("view_id") or path.stem)
        if view.get("is_overview"):
            description = f"视图 {view_id}: 完整等距柱状全景概览，保持 2:1。"
        else:
            description = (
                f"视图 {view_id}: yaw={view.get('yaw')}°, "
                f"pitch={view.get('pitch')}°, FOV={view.get('fov')}°, "
                f"尺寸={view.get('width')}x{view.get('height')}。"
            )
        content.append({"type": "text", "text": description})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_data_url(path)},
            }
        )
        valid_count += 1
    if not valid_count:
        raise ValueError("多图消息没有任何可读取的视图")
    return [SystemMessage(content=system), HumanMessage(content=content)]


def chat_with_images(
    system: str,
    user: str,
    views: list[Any],
    temperature: float = 0.2,
) -> Optional[str]:
    """通过现有 LangChain 模型客户端发送概览图和多张透视图。"""
    if use_mock_llm():
        return None
    try:
        messages = build_multi_image_messages(system, user, views)
        response = _build_model(temperature).invoke(messages)
        return _content_as_text(response.content) or None
    except Exception as exc:
        print(f"[LangChain] 多图场景理解失败，返回降级状态: {exc}")
        return None


__all__ = [
    "build_multi_image_messages",
    "chat",
    "chat_with_image",
    "chat_with_images",
]
