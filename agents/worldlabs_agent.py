"""
================================================================================
World Labs Pano Edit 调用 Agent
文件: agents/worldlabs_agent.py
--------------------------------------------------------------------------------
【角色】
  接收人工确认后的自然语言编辑指令 + 原始全景图，调用 World Labs API
  生成修改后的 360° 全景图。无 API Key 时进入 MOCK：在原图上叠加标注条，
  保证全流程可跑通。

【输入】
  - image_path: str           原始全景图路径
  - prompt: str               最终自然语言指令（提示词专家 + 人工润色）
  - output_name: str | None   输出文件名（可选）

【输出】
  - GenerationResult
      .output_image_path
      .prompt_used
      .mock
      .raw                    API 原始响应或 mock 元数据

【输出到哪里】
  → code/outputs/images/
  → 路径回传前端展示；随后交给质检员 Agent 重新解析形态要素

【怎么调用】
  from agents.worldlabs_agent import WorldLabsAgent
  agent = WorldLabsAgent()
  result = agent.run(image_path="uploads/pano.jpg", prompt="Add vertical greenery...")

【API 说明】
  World Labs Marble 以异步 Operation 为主。本封装优先尝试：
    POST {BASE}/marble/v1/worlds:generate  (image + text_prompt, is_pano=true)
  若官方 Pano Edit 专用端点变更，请在本文件 WORLDLABS_ENDPOINTS 中更新。
  当前文档侧重 Studio Pano Edit；API 可用性随账号配额变化，故默认支持 MOCK。
================================================================================
"""

from __future__ import annotations

import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    IMAGE_OUT_DIR,
    WORLDLABS_API_KEY,
    WORLDLABS_BASE_URL,
    use_mock_worldlabs,
)
from schemas.models import GenerationResult


WORLDLABS_ENDPOINTS = {
    "generate": "/marble/v1/worlds:generate",
    "operation": "/marble/v1/operations/{operation_id}",
    "prepare_upload": "/marble/v1/media:prepare_upload",
}


class WorldLabsAgent:
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = (api_key if api_key is not None else WORLDLABS_API_KEY).strip()
        self.base_url = (base_url or WORLDLABS_BASE_URL).rstrip("/")
        self.mock = use_mock_worldlabs() if api_key is None else (not bool(self.api_key))

    def run(
        self,
        image_path: str | Path,
        prompt: str,
        output_name: str | None = None,
        poll_timeout_sec: int = 300,
    ) -> GenerationResult:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        out_name = output_name or f"pano_edit_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg"
        out_path = IMAGE_OUT_DIR / out_name

        if self.mock:
            return self._run_mock(image_path, prompt, out_path)

        try:
            return self._run_live(image_path, prompt, out_path, poll_timeout_sec)
        except Exception as e:
            print(f"[WorldLabs] API 调用失败，回退 MOCK: {e}")
            result = self._run_mock(image_path, prompt, out_path)
            result.raw["fallback_error"] = str(e)
            return result

    # ------------------------------------------------------------------ #
    # LIVE
    # ------------------------------------------------------------------ #
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _run_live(
        self,
        image_path: Path,
        prompt: str,
        out_path: Path,
        poll_timeout_sec: int,
    ) -> GenerationResult:
        # 简化路径：许多账号以 worlds:generate + 本地图 URL/上传 完成；
        # 此处先 prepare_upload（若失败则直接带本地说明并报错触发 mock）
        media_uri = self._upload_image(image_path)

        payload = {
            "world_prompt": {
                "image_prompt": {
                    "image_uri": media_uri,
                    "text_prompt": prompt,
                    "is_pano": True,
                }
            }
        }
        url = f"{self.base_url}{WORLDLABS_ENDPOINTS['generate']}"
        resp = requests.post(url, headers=self._headers(), json=payload, timeout=60)
        resp.raise_for_status()
        op = resp.json()
        operation_id = op.get("operation_id") or op.get("name") or op.get("id")
        if not operation_id:
            raise RuntimeError(f"未返回 operation_id: {op}")

        done = self._poll_operation(operation_id, poll_timeout_sec)
        pano_url = self._extract_pano_url(done)
        if not pano_url:
            raise RuntimeError(f"操作完成但未找到全景 URL: {done}")

        img_bytes = requests.get(pano_url, timeout=120).content
        out_path.write_bytes(img_bytes)

        return GenerationResult(
            output_image_path=str(out_path),
            prompt_used=prompt,
            mock=False,
            raw={"operation": done, "pano_url": pano_url},
        )

    def _upload_image(self, image_path: Path) -> str:
        """尝试 prepare_upload；若端点不可用则抛错由上层 MOCK。"""
        url = f"{self.base_url}{WORLDLABS_ENDPOINTS['prepare_upload']}"
        meta = {
            "file_name": image_path.name,
            "content_type": "image/jpeg",
        }
        resp = requests.post(url, headers=self._headers(), json=meta, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        upload_url = data.get("upload_url") or data.get("put_url")
        media_uri = data.get("media_uri") or data.get("uri") or data.get("asset_uri")
        if upload_url:
            with open(image_path, "rb") as f:
                put = requests.put(
                    upload_url,
                    data=f,
                    headers={"Content-Type": "image/jpeg"},
                    timeout=120,
                )
                put.raise_for_status()
        if not media_uri:
            raise RuntimeError(f"prepare_upload 未返回 media_uri: {data}")
        return media_uri

    def _poll_operation(self, operation_id: str, timeout_sec: int) -> dict[str, Any]:
        url = f"{self.base_url}{WORLDLABS_ENDPOINTS['operation'].format(operation_id=operation_id)}"
        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            resp = requests.get(url, headers=self._headers(), timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if data.get("done") is True or data.get("status") in {"SUCCEEDED", "DONE", "completed"}:
                return data
            if data.get("status") in {"FAILED", "ERROR"}:
                raise RuntimeError(f"World Labs 操作失败: {data}")
            time.sleep(3)
        raise TimeoutError(f"轮询超时: {operation_id}")

    @staticmethod
    def _extract_pano_url(operation: dict[str, Any]) -> str | None:
        resp = operation.get("response") or operation
        for key in ("pano_url", "image_url", "url"):
            if resp.get(key):
                return resp[key]
        world = resp.get("world") or {}
        assets = world.get("assets") or resp.get("assets") or {}
        if isinstance(assets, dict):
            for key in ("pano_url", "panorama_url", "preview_url"):
                if assets.get(key):
                    return assets[key]
        return None

    # ------------------------------------------------------------------ #
    # MOCK：保证无密钥也能跑通前端闭环
    # ------------------------------------------------------------------ #
    @staticmethod
    def _imread_unicode(image_path: Path):
        from PIL import Image

        try:
            pil_img = Image.open(image_path).convert("RGB")
            rgb = np.array(pil_img)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            data = np.fromfile(str(image_path), dtype=np.uint8)
            return cv2.imdecode(data, cv2.IMREAD_COLOR)

    def _run_mock(self, image_path: Path, prompt: str, out_path: Path) -> GenerationResult:
        img = self._imread_unicode(image_path)
        if img is None:
            shutil.copy(image_path, out_path)
        else:
            h, w = img.shape[:2]
            overlay = img.copy()
            # 轻微绿色叠加，模拟“增加绿化”的视觉反馈
            green = np.zeros_like(img)
            green[:, :] = (40, 120, 40)
            cv2.addWeighted(overlay, 0.85, green, 0.15, 0, overlay)
            bar_h = max(40, h // 18)
            cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
            label = "MOCK WorldLabs Pano Edit | prompt applied"
            cv2.putText(
                overlay,
                label[:80],
                (20, int(bar_h * 0.7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (180, 255, 180),
                2,
                cv2.LINE_AA,
            )
            # 底部写提示词摘要
            summary = (prompt[:120] + "...") if len(prompt) > 120 else prompt
            cv2.rectangle(overlay, (0, h - bar_h), (w, h), (20, 20, 20), -1)
            cv2.putText(
                overlay,
                summary.encode("ascii", "ignore").decode() or "edit applied",
                (20, h - int(bar_h * 0.3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            # 兼容中文输出路径
            ok, buf = cv2.imencode(".jpg", overlay)
            if ok:
                buf.tofile(str(out_path))
            else:
                shutil.copy(image_path, out_path)

        return GenerationResult(
            output_image_path=str(out_path),
            prompt_used=prompt,
            mock=True,
            raw={"mode": "mock", "note": "未配置 WORLDLABS_API_KEY 或 API 失败，已生成本地模拟图"},
        )


def run_worldlabs(image_path: str, prompt: str) -> GenerationResult:
    return WorldLabsAgent().run(image_path, prompt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--prompt", default="Add cascading vertical garden on the right wall")
    args = parser.parse_args()
    r = WorldLabsAgent().run(args.image, args.prompt)
    print(r.model_dump_json(indent=2, ensure_ascii=False))
