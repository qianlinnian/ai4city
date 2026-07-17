"""临时联调脚本：读取 .env 中的 WORLDLABS_API_KEY，对 assets 中第一张 VID_ 图做真实文生图。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from agents.worldlabs_agent import WorldLabsAgent


def main() -> None:
    assets = ROOT / "assets"
    names = sorted(
        p.name
        for p in assets.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} and p.name.startswith("VID_")
    )
    if not names:
        raise SystemExit("assets/ 中没有 VID_ 开头的图片")

    name = names[0]
    print("using_image:", name)
    agent = WorldLabsAgent()
    print("mock_mode:", agent.mock)
    if agent.mock:
        raise SystemExit("当前仍是 MOCK，请确认 .env 中 WORLDLABS_API_KEY 与 RUN_MODE=live")

    result = agent.run(
        image_name=name,
        prompt=(
            "Slightly increase greenery along the path edges, "
            "keep the original urban street layout and camera viewpoint."
        ),
        poll_timeout_sec=600,
        allow_mock_fallback=False,
    )
    print("SUCCESS")
    print("mock:", result.mock)
    print("out:", result.output_image_path)
    out = Path(result.output_image_path)
    print("exists:", out.exists(), "size:", out.stat().st_size if out.exists() else 0)
    raw = result.raw or {}
    print("world_id:", raw.get("world_id"))
    print("operation_id:", raw.get("operation_id"))
    print("upload_mode:", raw.get("upload_mode"))
    print("pano_url_prefix:", (raw.get("pano_url") or "")[:100])


if __name__ == "__main__":
    main()
