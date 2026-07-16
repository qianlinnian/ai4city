"""
LLM 客户端（OpenAI 兼容）
无 API Key 时返回 None，由各 Agent 走规则/模板 MOCK 路径。
"""

from __future__ import annotations

from typing import Optional

import requests

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, use_mock_llm


def chat(system: str, user: str, temperature: float = 0.4) -> Optional[str]:
    """
    调用聊天补全。
    返回: 文本；MOCK 或失败时返回 None（调用方应使用本地规则兜底）
    """
    if use_mock_llm():
        return None

    url = f"{LLM_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=90)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[LLM] 调用失败，回退本地规则: {e}")
        return None
