"""
================================================================================
World Labs Marble 文生图工具(Pano / Image-to-World)
文件: agents/worldlabs_agent.py
--------------------------------------------------------------------------------
【角色】
  根据「图片完整文件名」从 assets/ 读取原始全景图，结合自然语言修改指令，
  调用 World Labs Marble API 生成优化后的全景图，保存到 TargetIMG/。

【官方 API 依据】
  https://docs.worldlabs.ai/api
  - 鉴权头: WLT-Api-Key
  - 上传:   POST /marble/v1/media-assets:prepare_upload → PUT 签名 URL
  - 生成:   POST /marble/v1/worlds:generate  (type=image, is_pano=true)
  - 轮询:   GET  /marble/v1/operations/{operation_id}
  - 结果:   response.assets.imagery.pano_url

【输入】
  - image_name: str   图片完整文件名（在 assets/ 下按文件名查找）
                      例: "VID_20260707_161752_00_144_2026-07-16_14-56-42_截图.jpg"
  - prompt: str       最终自然语言修改指令
  兼容：
  - image_path        直接指定本地绝对/相对路径（编排器用）
  - image_id          旧参数名，等同于 image_name

【输出】
  - GenerationResult
      .output_image_path   TargetIMG/ 下生成图路径
      .prompt_used
      .mock                是否 MOCK
      .raw                 API 原始响应 / world_id / pano_url

【输出到哪里】
  → code/TargetIMG/{原文件名stem}_edited_{timestamp}.jpg

【怎么调用】
  from agents.worldlabs_agent import WorldLabsAgent
  agent = WorldLabsAgent()

  result = agent.run(
      image_name="VID_20260707_161752_00_144_2026-07-16_14-56-42_截图.jpg",
      prompt="Add cascading vertical garden on the right wall",
  )

  # 命令行
  python agents/worldlabs_agent.py --image-name "VID_xxx_截图.jpg" --prompt "Add trees"
================================================================================
"""

from __future__ import annotations

import base64
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    ASSETS_DIR,
    TARGET_IMG_DIR,
    WORLDLABS_API_KEY,
    WORLDLABS_BASE_URL,
    WORLDLABS_MODEL,
    use_mock_worldlabs,
)
from schemas.models import GenerationResult


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP")

WORLDLABS_ENDPOINTS = {
    "prepare_upload": "/marble/v1/media-assets:prepare_upload",
    "generate": "/marble/v1/worlds:generate",
    "operation": "/marble/v1/operations/{operation_id}",
    "world": "/marble/v1/worlds/{world_id}",
}


class WorldLabsAgent:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        assets_dir: Path | str | None = None,
        target_dir: Path | str | None = None,
        model: str | None = None,
    ):
        self.api_key = (api_key if api_key is not None else WORLDLABS_API_KEY).strip()
        self.base_url = (base_url or WORLDLABS_BASE_URL).rstrip("/")
        self.assets_dir = Path(assets_dir or ASSETS_DIR)
        self.target_dir = Path(target_dir or TARGET_IMG_DIR)
        self.model = model or WORLDLABS_MODEL
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.target_dir.mkdir(parents=True, exist_ok=True)
        # 显式传入 api_key 时以是否为空判断；否则跟随 RUN_MODE
        if api_key is not None:
            self.mock = not bool(self.api_key)
        else:
            self.mock = use_mock_worldlabs()

    # ------------------------------------------------------------------ #
    # 对外主入口
    # ------------------------------------------------------------------ #
    def run(
        self,
        image_name: str | Path | None = None,
        prompt: str = "",
        *,
        image_path: str | Path | None = None,
        image_id: str | Path | None = None,
        output_name: str | None = None,
        poll_timeout_sec: int = 600,
        poll_interval_sec: float = 5.0,
        is_pano: bool | str = True,
        disable_recaption: bool = True,
        display_name: str | None = None,
        allow_mock_fallback: bool = True,
    ) -> GenerationResult:
        """
        文生图主入口。

        参数优先级：
          1) 显式 image_path
          2) image_name / image_id 若本身是已存在路径 → 当路径用
          3) 在 assets/ 中按「完整文件名」查找
        """
        if not prompt or not str(prompt).strip():
            raise ValueError("prompt（自然语言指令）不能为空")

        # image_id 为旧别名，优先使用 image_name
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
                image_id=resolved_name,
                poll_timeout_sec=poll_timeout_sec,
                poll_interval_sec=poll_interval_sec,
                is_pano=is_pano,
                disable_recaption=disable_recaption,
                display_name=display_name or f"pano_edit_{Path(resolved_name).stem[:40]}",
            )
        except Exception as e:
            print(f"[WorldLabs] API 调用失败: {e}")
            if not allow_mock_fallback:
                raise
            print("[WorldLabs] 回退 MOCK 本地演示图")
            result = self._run_mock(resolved_path, prompt, out_path, resolved_name)
            result.raw["fallback_error"] = str(e)
            return result

    def resolve_image_name(self, image_name: str) -> Path:
        """根据图片完整文件名在 assets/ 查找路径。"""
        return self._find_in_assets(str(image_name).strip())

    # 兼容旧名
    def resolve_image_id(self, image_id: str) -> Path:
        return self.resolve_image_name(image_id)

    # ------------------------------------------------------------------ #
    # 路径解析
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

        # 兼容：传入已存在的本地路径
        as_path = Path(image_name)
        if as_path.exists() and as_path.is_file():
            return as_path.name, as_path
        if any(sep in str(image_name) for sep in ("/", "\\")) and as_path.suffix:
            return as_path.name, as_path

        name = str(image_name).strip()
        # 若带路径前缀，只取文件名部分去 assets 查
        name = Path(name).name
        found = self._find_in_assets(name)
        return found.name, found

    def _find_in_assets(self, image_name: str) -> Path:
        """
        在 assets/ 中按图片名称查找。匹配顺序：
          1) 完整文件名精确匹配（推荐）
             例: VID_20260707_161752_00_144_2026-07-16_14-56-42_截图.jpg
          2) 大小写不敏感的完整文件名匹配
          3) 未带扩展名时，尝试补全 .jpg/.png/...
          4) 仅 stem 匹配（兼容旧的短 ID，如 scene_001）
        """
        if not image_name:
            raise ValueError("image_name 为空")

        if not self.assets_dir.exists():
            raise FileNotFoundError(f"assets 目录不存在: {self.assets_dir}")

        # 1) 完整文件名精确匹配
        exact_path = self.assets_dir / image_name
        if exact_path.is_file():
            return exact_path

        files = [
            p for p in self.assets_dir.iterdir()
            if p.is_file() and p.suffix in IMAGE_EXTENSIONS
        ]

        # 2) 大小写不敏感
        lower_name = image_name.lower()
        for p in files:
            if p.name.lower() == lower_name:
                return p

        # 3) 未带扩展名 → 补全扩展名
        name_path = Path(image_name)
        if not name_path.suffix:
            for ext in IMAGE_EXTENSIONS:
                candidate = self.assets_dir / f"{image_name}{ext}"
                if candidate.is_file():
                    return candidate
            for p in files:
                if p.stem == image_name or p.stem.lower() == image_name.lower():
                    return p

        # 4) stem 精确匹配（仅当传入的是无扩展名或不完整时的兜底）
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
            f"并用完整文件名调用，例如:\n"
            f'  image_name="VID_20260707_161752_00_144_2026-07-16_14-56-42_截图.jpg"\n'
            f"当前可用文件: {available or '(空)'}"
        )

    @staticmethod
    def _default_output_name(image_name: str, source_path: Path) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = source_path.suffix.lower() if source_path.suffix else ".jpg"
        if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
            ext = ".jpg"
        stem = Path(image_name).stem if image_name else source_path.stem
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:60]
        return f"{safe}_edited_{ts}{ext}"

    # ------------------------------------------------------------------ #
    # LIVE API（官方 Marble）
    # ------------------------------------------------------------------ #
    def _headers(self, json_body: bool = True) -> dict[str, str]:
        headers = {"WLT-Api-Key": self.api_key}
        if json_body:
            headers["Content-Type"] = "application/json"
        return headers

    def _run_live(
        self,
        image_path: Path,
        prompt: str,
        out_path: Path,
        image_id: str,
        poll_timeout_sec: int,
        poll_interval_sec: float,
        is_pano: bool | str,
        disable_recaption: bool,
        display_name: str,
    ) -> GenerationResult:
        # 官方校验：is_pano 只能是 "auto" / True / False（布尔，不能是字符串 "true"）
        pano_mode: bool | str
        if isinstance(is_pano, bool):
            pano_mode = is_pano
        else:
            s = str(is_pano).strip().lower()
            if s == "auto":
                pano_mode = "auto"
            elif s in {"true", "1", "yes"}:
                pano_mode = True
            elif s in {"false", "0", "no"}:
                pano_mode = False
            else:
                pano_mode = True

        media_asset_id: str | None = None
        upload_mode = "media_asset"
        try:
            media_asset_id = self._upload_image(image_path)
            print(f"[WorldLabs] 已上传 media_asset_id={media_asset_id}")
            payload = {
                "display_name": display_name[:64],
                "model": self.model,
                "world_prompt": {
                    "type": "image",
                    "image_prompt": {
                        "source": "media_asset",
                        "media_asset_id": media_asset_id,
                    },
                    "text_prompt": prompt,
                    "is_pano": pano_mode,
                    "disable_recaption": disable_recaption,
                },
            }
            url = f"{self.base_url}{WORLDLABS_ENDPOINTS['generate']}"
            resp = requests.post(url, headers=self._headers(), json=payload, timeout=120)
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"worlds:generate HTTP {resp.status_code}: {resp.text[:800]}"
                )
            op = resp.json()
        except Exception as upload_err:
            print(f"[WorldLabs] media_asset 路径失败，改用 data_base64: {upload_err}")
            upload_mode = "data_base64"
            op = self._generate_with_base64(
                image_path, prompt, display_name, pano_mode, disable_recaption
            )

        operation_id = op.get("operation_id") or op.get("name") or op.get("id")
        if not operation_id:
            raise RuntimeError(f"未返回 operation_id: {op}")

        print(f"[WorldLabs] 生成已提交 operation_id={operation_id}，开始轮询…")
        done = self._poll_operation(operation_id, poll_timeout_sec, poll_interval_sec)

        if done.get("error"):
            raise RuntimeError(f"World Labs 操作失败: {done.get('error')}")

        pano_url = self._extract_pano_url(done)
        world_id = self._extract_world_id(done)

        if not pano_url and world_id:
            print(f"[WorldLabs] 操作完成但无 pano_url，尝试 GET world={world_id}")
            world = self._get_world(world_id)
            pano_url = self._extract_pano_url({"response": world})
            done["fetched_world"] = world

        if not pano_url:
            raise RuntimeError(f"操作完成但未找到全景 pano_url: {done}")

        print(f"[WorldLabs] 下载全景: {pano_url[:120]}...")
        img_bytes = requests.get(pano_url, timeout=180).content
        if len(img_bytes) < 1000:
            raise RuntimeError("下载的全景图过小，可能无效")
        out_path.write_bytes(img_bytes)
        print(f"[WorldLabs] 已保存: {out_path}")

        return GenerationResult(
            output_image_path=str(out_path),
            prompt_used=prompt,
            mock=False,
            raw={
                "image_id": image_id,
                "upload_mode": upload_mode,
                "media_asset_id": media_asset_id,
                "operation_id": operation_id,
                "world_id": world_id,
                "pano_url": pano_url,
                "model": self.model,
                "operation": done,
            },
        )

    def _upload_image(self, image_path: Path) -> str:
        """
        官方三步：prepare_upload → PUT 文件 → 返回 media_asset_id。
        file_name 最长 64 字符。
        """
        ext = image_path.suffix.lstrip(".").lower() or "jpg"
        if ext == "jpeg":
            ext = "jpg"
        # file_name maxLength=64
        safe_stem = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in image_path.stem
        )[:40] or "pano"
        file_name = f"{safe_stem}.{ext}"[:64]

        url = f"{self.base_url}{WORLDLABS_ENDPOINTS['prepare_upload']}"
        meta = {
            "file_name": file_name,
            "kind": "image",
            "extension": ext,
        }
        resp = requests.post(url, headers=self._headers(), json=meta, timeout=60)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"prepare_upload HTTP {resp.status_code}: {resp.text[:500]}"
            )

        data = resp.json()
        media_asset = data.get("media_asset") or {}
        media_asset_id = (
            media_asset.get("media_asset_id")
            or media_asset.get("id")
            or data.get("media_asset_id")
        )
        upload_info = data.get("upload_info") or {}
        upload_url = upload_info.get("upload_url") or data.get("upload_url")
        required_headers = dict(upload_info.get("required_headers") or {})

        if not media_asset_id:
            raise RuntimeError(f"prepare_upload 未返回 media_asset_id: {data}")
        if not upload_url:
            raise RuntimeError(f"prepare_upload 未返回 upload_url: {data}")

        content_type = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "webp": "image/webp",
        }.get(ext, "image/jpeg")
        put_headers = {"Content-Type": content_type, **required_headers}

        with open(image_path, "rb") as f:
            put = requests.put(upload_url, data=f, headers=put_headers, timeout=300)
        if put.status_code >= 400:
            raise RuntimeError(
                f"上传图片失败 HTTP {put.status_code}: {put.text[:500]}"
            )
        return str(media_asset_id)

    def _generate_with_base64(
        self,
        image_path: Path,
        prompt: str,
        display_name: str,
        is_pano: bool | str,
        disable_recaption: bool,
    ) -> dict[str, Any]:
        """不经 media_asset，直接用 data_base64 提交 worlds:generate。"""
        ext = image_path.suffix.lstrip(".").lower() or "jpg"
        if ext == "jpeg":
            ext = "jpg"
        raw = image_path.read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        payload = {
            "display_name": display_name[:64],
            "model": self.model,
            "world_prompt": {
                "type": "image",
                "image_prompt": {
                    "source": "data_base64",
                    "data_base64": b64,
                    "extension": ext,
                },
                "text_prompt": prompt,
                "is_pano": is_pano,
                "disable_recaption": disable_recaption,
            },
        }
        url = f"{self.base_url}{WORLDLABS_ENDPOINTS['generate']}"
        resp = requests.post(url, headers=self._headers(), json=payload, timeout=180)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"worlds:generate(base64) HTTP {resp.status_code}: {resp.text[:800]}"
            )
        return resp.json()

    def _poll_operation(
        self,
        operation_id: str,
        timeout_sec: int,
        interval_sec: float,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{WORLDLABS_ENDPOINTS['operation'].format(operation_id=operation_id)}"
        t0 = time.time()
        last_status = None
        consecutive_net_errors = 0
        while time.time() - t0 < timeout_sec:
            try:
                resp = requests.get(url, headers=self._headers(json_body=False), timeout=60)
                consecutive_net_errors = 0
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                consecutive_net_errors += 1
                print(f"[WorldLabs] 轮询网络异常({consecutive_net_errors}): {e}")
                if consecutive_net_errors >= 20:
                    raise
                time.sleep(min(30.0, interval_sec * consecutive_net_errors))
                continue

            if resp.status_code >= 400:
                raise RuntimeError(
                    f"轮询 operation 失败 HTTP {resp.status_code}: {resp.text[:500]}"
                )
            data = resp.json()
            if data.get("done") is True:
                return data
            if data.get("error"):
                raise RuntimeError(f"World Labs 操作失败: {data['error']}")

            meta = data.get("metadata") or {}
            progress = meta.get("progress") or {}
            status = progress.get("status") or data.get("status") or "IN_PROGRESS"
            if status != last_status:
                desc = progress.get("description") or ""
                print(f"[WorldLabs] 状态: {status} {desc}")
                last_status = status
            if status in {"FAILED", "ERROR"}:
                raise RuntimeError(f"World Labs 操作失败: {data}")
            time.sleep(interval_sec)
        raise TimeoutError(f"轮询超时 ({timeout_sec}s): {operation_id}")

    def _get_world(self, world_id: str) -> dict[str, Any]:
        url = f"{self.base_url}{WORLDLABS_ENDPOINTS['world'].format(world_id=world_id)}"
        resp = requests.get(url, headers=self._headers(json_body=False), timeout=60)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _extract_world_id(operation: dict[str, Any]) -> str | None:
        meta = operation.get("metadata") or {}
        if meta.get("world_id"):
            return meta["world_id"]
        resp = operation.get("response") or {}
        return resp.get("id") or resp.get("world_id")

    @staticmethod
    def _extract_pano_url(operation: dict[str, Any]) -> str | None:
        resp = operation.get("response") or operation
        if not isinstance(resp, dict):
            return None

        # 官方路径: response.assets.imagery.pano_url
        assets = resp.get("assets") or {}
        if isinstance(assets, dict):
            imagery = assets.get("imagery") or {}
            if isinstance(imagery, dict) and imagery.get("pano_url"):
                return imagery["pano_url"]
            for key in ("pano_url", "panorama_url", "preview_url", "thumbnail_url"):
                if assets.get(key):
                    return assets[key]

        for key in ("pano_url", "image_url", "url"):
            if resp.get(key):
                return resp[key]
        return None

    # ------------------------------------------------------------------ #
    # MOCK：无密钥时保证流水线可跑通
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
        image_id: str,
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
            label = f"MOCK WorldLabs | {image_id[:40]}"
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
                "image_id": image_id,
                "note": "未配置 WORLDLABS_API_KEY 或 API 失败，已生成本地模拟图到 TargetIMG/",
            },
        )


def run_worldlabs(
    image_name: str | None = None,
    prompt: str = "",
    *,
    image_path: str | None = None,
    image_id: str | None = None,
) -> GenerationResult:
    return WorldLabsAgent().run(
        image_name=image_name or image_id,
        prompt=prompt,
        image_path=image_path,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="World Labs Marble 全景文生图工具")
    parser.add_argument(
        "--image-name",
        help='assets/ 中的完整图片文件名，如 "VID_20260707_161752_00_144_2026-07-16_14-56-42_截图.jpg"',
    )
    parser.add_argument("--image-id", help="兼容旧参数，等同于 --image-name")
    parser.add_argument("--image", help="直接指定本地图片路径（兼容）")
    parser.add_argument("--prompt", required=True, help="自然语言修改指令")
    parser.add_argument("--no-mock-fallback", action="store_true", help="API 失败不回退 MOCK")
    parser.add_argument("--timeout", type=int, default=600, help="轮询超时秒数")
    args = parser.parse_args()

    name = args.image_name or args.image_id
    if not name and not args.image:
        parser.error("请提供 --image-name 或 --image")

    agent = WorldLabsAgent()
    r = agent.run(
        image_name=name,
        prompt=args.prompt,
        image_path=args.image,
        poll_timeout_sec=args.timeout,
        allow_mock_fallback=not args.no_mock_fallback,
    )
    print(r.model_dump_json(indent=2, ensure_ascii=False))
