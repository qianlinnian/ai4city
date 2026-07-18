"""
================================================================================
豆包 Seedream 文生图工具（图生图 / Image-to-Image）
文件: agents/seedream_agent.py
--------------------------------------------------------------------------------
【角色】
  根据「图片完整文件名」从 assets/ 读取原始全景图，结合自然语言修改指令，
  调用火山方舟 Seedream API 生成改造后的全景图，保存到 TargetIMG/。

【官方 API 依据】
  POST https://ark.cn-beijing.volces.com/api/v3/images/generations
  - 鉴权: Authorization: Bearer {ARK_API_KEY}
  - 图生图: 传入 prompt + image（URL 或 base64 data URI）
  - 模型: doubao-seedream-5-0-260128（Seedream 5.0 lite）

【输入】
  - image_name: str   图片完整文件名（在 assets/ 下按文件名查找）
  - prompt: str       最终自然语言修改指令
  兼容：
  - image_path        直接指定本地路径（编排器用）
  - image_id          旧参数名，等同于 image_name

【输出】
  - GenerationResult
      .output_image_path   TargetIMG/ 下生成图路径
      .prompt_used
      .mock                是否 MOCK
      .raw                 API 原始响应

【输出到哪里】
  → code/TargetIMG/{原文件名stem}_edited_{timestamp}.jpg

【怎么调用】
  from agents.seedream_agent import SeedreamAgent
  agent = SeedreamAgent()
  result = agent.run(
      image_name="VID_20260707_161752_00_144_2026-07-16_14-56-42_截图.jpg",
      prompt="Add lush street trees and vertical greening on building facades",
  )

  # 命令行
  python agents/seedream_agent.py --image-name "VID_xxx_截图.jpg" --prompt "..."
================================================================================
"""

from __future__ import annotations

import base64
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    ASSETS_DIR,
    SEEDREAM_API_KEY,
    SEEDREAM_BASE_URL,
    SEEDREAM_MODEL,
    SEEDREAM_RESPONSE_FORMAT,
    SEEDREAM_SIZE,
    SEEDREAM_WATERMARK,
    TARGET_IMG_DIR,
    use_mock_seedream,
)
from schemas.models import GenerationResult


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP")

SEEDREAM_ENDPOINT = "/api/v3/images/generations"


class SeedreamAgent:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        assets_dir: Path | str | None = None,
        target_dir: Path | str | None = None,
        model: str | None = None,
        size: str | None = None,
        watermark: bool | None = None,
        response_format: str | None = None,
    ):
        self.api_key = (api_key if api_key is not None else SEEDREAM_API_KEY).strip()
        self.base_url = (base_url or SEEDREAM_BASE_URL).rstrip("/")
        self.assets_dir = Path(assets_dir or ASSETS_DIR)
        self.target_dir = Path(target_dir or TARGET_IMG_DIR)
        self.model = model or SEEDREAM_MODEL
        self.size = size or SEEDREAM_SIZE
        self.watermark = SEEDREAM_WATERMARK if watermark is None else watermark
        self.response_format = response_format or SEEDREAM_RESPONSE_FORMAT
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.target_dir.mkdir(parents=True, exist_ok=True)
        if api_key is not None:
            self.mock = not bool(self.api_key)
        else:
            self.mock = use_mock_seedream()

    def run(
        self,
        image_name: str | Path | None = None,
        prompt: str = "",
        *,
        image_path: str | Path | None = None,
        image_id: str | Path | None = None,
        output_name: str | None = None,
        size: str | None = None,
        watermark: bool | None = None,
        response_format: str | None = None,
        allow_mock_fallback: bool = True,
        # 以下参数仅为与 WorldLabsAgent 接口兼容，Seedream 不使用
        poll_timeout_sec: int = 600,
        poll_interval_sec: float = 5.0,
        is_pano: bool | str = True,
        disable_recaption: bool = True,
        display_name: str | None = None,
    ) -> GenerationResult:
        """
        文生图主入口（图生图）。

        参数优先级：
          1) 显式 image_path
          2) image_name / image_id 若本身是已存在路径 → 当路径用
          3) 在 assets/ 中按「完整文件名」查找
        """
        if not prompt or not str(prompt).strip():
            raise ValueError("prompt（自然语言指令）不能为空")

        name_or_id = image_name if image_name is not None else image_id
        resolved_name, resolved_path = self._resolve_input(name_or_id, image_path)
        if not resolved_path.exists():
            raise FileNotFoundError(
                f"找不到输入图片: name={resolved_name!r}, path={resolved_path}"
            )

        out_name = output_name or self._default_output_name(resolved_name, resolved_path)
        out_path = self.target_dir / out_name

        if self.mock:
            return self._run_mock(resolved_path, prompt, out_path, resolved_name)

        try:
            return self._run_live(
                image_path=resolved_path,
                prompt=prompt.strip(),
                out_path=out_path,
                image_name=resolved_name,
                size=size or self.size,
                watermark=self.watermark if watermark is None else watermark,
                response_format=response_format or self.response_format,
            )
        except Exception as e:
            print(f"[Seedream] API 调用失败: {e}")
            if not allow_mock_fallback:
                raise
            print("[Seedream] 回退 MOCK 本地演示图")
            result = self._run_mock(resolved_path, prompt, out_path, resolved_name)
            result.raw["fallback_error"] = str(e)
            return result

    def resolve_image_name(self, image_name: str) -> Path:
        """根据图片完整文件名在 assets/ 查找路径。"""
        return self._find_in_assets(str(image_name).strip())

    def resolve_image_id(self, image_id: str) -> Path:
        return self.resolve_image_name(image_id)

    # ------------------------------------------------------------------ #
    # 路径解析（与 worldlabs_agent 保持一致）
    # ------------------------------------------------------------------ #
    def _resolve_input(
        self,
        image_name: str | Path | None,
        image_path: str | Path | None,
    ) -> tuple[str, Path]:
        if image_path is not None:
            path = Path(image_path)
            return path.name, path

        if image_name is None:
            raise ValueError("必须提供 image_name（图片文件名）或 image_path")

        as_path = Path(image_name)
        if as_path.exists() and as_path.is_file():
            return as_path.name, as_path
        if any(sep in str(image_name) for sep in ("/", "\\")) and as_path.suffix:
            return as_path.name, as_path

        name = Path(str(image_name).strip()).name
        found = self._find_in_assets(name)
        return found.name, found

    def _find_in_assets(self, image_name: str) -> Path:
        if not image_name:
            raise ValueError("image_name 为空")
        if not self.assets_dir.exists():
            raise FileNotFoundError(f"assets 目录不存在: {self.assets_dir}")

        exact_path = self.assets_dir / image_name
        if exact_path.is_file():
            return exact_path

        files = [
            p for p in self.assets_dir.iterdir()
            if p.is_file() and p.suffix in IMAGE_EXTENSIONS
        ]

        lower_name = image_name.lower()
        for p in files:
            if p.name.lower() == lower_name:
                return p

        name_path = Path(image_name)
        if not name_path.suffix:
            for ext in IMAGE_EXTENSIONS:
                candidate = self.assets_dir / f"{image_name}{ext}"
                if candidate.is_file():
                    return candidate
            for p in files:
                if p.stem == image_name or p.stem.lower() == image_name.lower():
                    return p

        stem = name_path.stem if name_path.suffix else image_name
        stem_hits = [p for p in files if p.stem == stem]
        if len(stem_hits) == 1:
            return stem_hits[0]
        if len(stem_hits) > 1:
            names = ", ".join(p.name for p in stem_hits)
            raise FileNotFoundError(
                f"assets/ 中有多个文件 stem={stem!r}，请传入完整文件名。候选: {names}"
            )

        available = sorted(p.name for p in files)[:30]
        raise FileNotFoundError(
            f"assets/ 中未找到图片名称 {image_name!r}。\n"
            f"请把图片放到: {self.assets_dir}\n"
            f"当前可用文件: {available or '(空)'}"
        )

    @staticmethod
    def _default_output_name(image_name: str, source_path: Path) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 统一用 .jpg，避免中文文件名导致 Windows/OpenCV/Gradio 打不开
        stem = Path(image_name).stem if image_name else source_path.stem
        # 仅保留 ASCII，中文（如「截图」）在 Python 里 isalnum=True，必须额外过滤
        safe = "".join(c if c.isascii() and (c.isalnum() or c in "-_") else "_" for c in stem)
        safe = "_".join(p for p in safe.split("_") if p)[:60] or "pano"
        return f"{safe}_edited_{ts}.jpg"

    # ------------------------------------------------------------------ #
    # LIVE API（火山方舟 Seedream）
    # ------------------------------------------------------------------ #
    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    @staticmethod
    def _prepare_image_bytes(
        image_path: Path,
        max_bytes: int = 9 * 1024 * 1024,
    ) -> tuple[bytes, str]:
        """
        准备上传字节。超过 max_bytes 时自动 JPEG 压缩/缩小，
        避免 Seedream 10MB 限制导致误回退 MOCK。
        返回 (bytes, mime_subtype) 如 (..., \"jpeg\")。
        """
        from io import BytesIO

        from PIL import Image

        raw = image_path.read_bytes()
        ext = image_path.suffix.lstrip(".").lower() or "jpg"
        mime_ext = "jpeg" if ext in {"jpg", "jpeg"} else ext
        if mime_ext not in {"jpeg", "png", "webp"}:
            mime_ext = "jpeg"

        if len(raw) <= max_bytes and mime_ext in {"jpeg", "png", "webp"}:
            return raw, mime_ext

        print(
            f"[Seedream] 原图 {len(raw)/1024/1024:.1f}MB 超限，自动压缩后上传 "
            f"(源文件: {image_path.name})"
        )
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            # 全景常见很大，先限制长边
            max_side = 4096
            w, h = im.size
            scale = min(1.0, max_side / max(w, h))
            if scale < 1.0:
                im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)

            for quality in (90, 85, 80, 75, 70, 65, 60, 50, 40):
                buf = BytesIO()
                im.save(buf, format="JPEG", quality=quality, optimize=True)
                data = buf.getvalue()
                if len(data) <= max_bytes:
                    print(
                        f"[Seedream] 压缩完成: {len(data)/1024/1024:.1f}MB "
                        f"(quality={quality}, size={im.size[0]}x{im.size[1]})"
                    )
                    return data, "jpeg"

            # 仍过大则继续缩小边长
            while True:
                w, h = im.size
                if max(w, h) <= 1024:
                    break
                im = im.resize((max(1, w // 2), max(1, h // 2)), Image.Resampling.LANCZOS)
                buf = BytesIO()
                im.save(buf, format="JPEG", quality=70, optimize=True)
                data = buf.getvalue()
                if len(data) <= max_bytes:
                    print(
                        f"[Seedream] 二次缩小完成: {len(data)/1024/1024:.1f}MB "
                        f"(size={im.size[0]}x{im.size[1]})"
                    )
                    return data, "jpeg"

        raise ValueError(
            f"无法将图片压缩到 ≤{max_bytes/1024/1024:.0f}MB: {image_path.name}"
        )

    @classmethod
    def _image_to_data_uri(cls, image_path: Path) -> str:
        raw, mime_ext = cls._prepare_image_bytes(image_path)
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:image/{mime_ext};base64,{b64}"

    @staticmethod
    def _resolve_size(image_path: Path, size: str) -> str:
        """若 size=auto，按原图宽高比给出接近 2K 的像素尺寸。"""
        if size.lower() != "auto":
            return size
        try:
            from PIL import Image

            with Image.open(image_path) as im:
                w, h = im.size
            if w <= 0 or h <= 0:
                return "2K"
            target_pixels = 2048 * 2048
            scale = (target_pixels / (w * h)) ** 0.5
            nw = max(512, int(w * scale))
            nh = max(512, int(h * scale))
            # 官方要求总像素与宽高比在合法区间，全景常用 2:1
            return f"{nw}x{nh}"
        except Exception:
            return "2K"

    def _run_live(
        self,
        image_path: Path,
        prompt: str,
        out_path: Path,
        image_name: str,
        size: str,
        watermark: bool,
        response_format: str,
    ) -> GenerationResult:
        image_ref = self._image_to_data_uri(image_path)
        resolved_size = self._resolve_size(image_path, size)

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "image": image_ref,
            "sequential_image_generation": "disabled",
            "response_format": response_format,
            "size": resolved_size,
            "stream": False,
            "watermark": watermark,
        }

        url = f"{self.base_url}{SEEDREAM_ENDPOINT}"
        print(f"[Seedream] 请求 model={self.model}, size={resolved_size}")
        resp = requests.post(url, headers=self._headers(), json=payload, timeout=300)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"images/generations HTTP {resp.status_code}: {resp.text[:1000]}"
            )

        data = resp.json()
        items = data.get("data") or []
        if not items:
            raise RuntimeError(f"API 未返回图片 data: {data}")

        first = items[0]
        if response_format == "b64_json":
            b64_text = first.get("b64_json")
            if not b64_text:
                raise RuntimeError(f"response_format=b64_json 但无 b64_json 字段: {first}")
            img_bytes = base64.b64decode(b64_text)
            out_path.write_bytes(img_bytes)
            image_url = None
        else:
            image_url = first.get("url")
            if not image_url:
                raise RuntimeError(f"response_format=url 但无 url 字段: {first}")
            img_bytes = requests.get(image_url, timeout=180).content
            if len(img_bytes) < 1000:
                raise RuntimeError("下载的图片过小，可能无效")
            out_path.write_bytes(img_bytes)

        print(f"[Seedream] 已保存: {out_path}")

        return GenerationResult(
            output_image_path=str(out_path),
            prompt_used=prompt,
            mock=False,
            raw={
                "image_name": image_name,
                "model": self.model,
                "size": resolved_size,
                "image_url": image_url,
                "response": data,
            },
        )

    # ------------------------------------------------------------------ #
    # MOCK
    # ------------------------------------------------------------------ #
    @staticmethod
    def _imread_unicode(image_path: Path):
        import cv2
        import numpy as np
        from PIL import Image

        try:
            pil_img = Image.open(image_path).convert("RGB")
            rgb = np.array(pil_img)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            data = np.fromfile(str(image_path), dtype=np.uint8)
            return cv2.imdecode(data, cv2.IMREAD_COLOR)

    def _run_mock(
        self,
        image_path: Path,
        prompt: str,
        out_path: Path,
        image_name: str,
    ) -> GenerationResult:
        import cv2
        import numpy as np

        img = self._imread_unicode(image_path)
        if img is None:
            shutil.copy(image_path, out_path)
        else:
            h, w = img.shape[:2]
            overlay = img.copy()
            green = np.zeros_like(img)
            green[:, :] = (40, 120, 40)
            cv2.addWeighted(overlay, 0.85, green, 0.15, 0, overlay)
            bar_h = max(40, h // 18)
            cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
            label = f"MOCK Seedream | {image_name[:40]}"
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
            ok, buf = cv2.imencode(".jpg", overlay)
            if ok:
                buf.tofile(str(out_path))
            else:
                shutil.copy(image_path, out_path)

        return GenerationResult(
            output_image_path=str(out_path),
            prompt_used=prompt,
            mock=True,
            raw={
                "mode": "mock",
                "image_name": image_name,
                "note": "未配置 SEEDREAM_API_KEY 或 API 失败，已生成本地模拟图到 TargetIMG/",
            },
        )


def run_seedream(
    image_name: str | None = None,
    prompt: str = "",
    *,
    image_path: str | None = None,
    image_id: str | None = None,
) -> GenerationResult:
    return SeedreamAgent().run(
        image_name=image_name or image_id,
        prompt=prompt,
        image_path=image_path,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="豆包 Seedream 全景图生图工具")
    parser.add_argument(
        "--image-name",
        help='assets/ 中的完整图片文件名',
    )
    parser.add_argument("--image-id", help="兼容旧参数，等同于 --image-name")
    parser.add_argument("--image", help="直接指定本地图片路径")
    parser.add_argument("--prompt", required=True, help="自然语言修改指令")
    parser.add_argument("--size", default=None, help="输出尺寸，如 2K / 2048x1024 / auto")
    parser.add_argument("--no-mock-fallback", action="store_true", help="API 失败不回退 MOCK")
    args = parser.parse_args()

    name = args.image_name or args.image_id
    if not name and not args.image:
        parser.error("请提供 --image-name 或 --image")

    agent = SeedreamAgent()
    r = agent.run(
        image_name=name,
        prompt=args.prompt,
        image_path=args.image,
        size=args.size,
        allow_mock_fallback=not args.no_mock_fallback,
    )
    print(r.model_dump_json(indent=2, ensure_ascii=False))
