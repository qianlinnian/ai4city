"""
一键演示入口 v2
用法:
  python run_demo.py path/to/pano.jpg
  python run_demo.py   # 自动找 ../JPG素材 下第一张图
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline.orchestrator import run_full_demo


def find_sample() -> Path:
    parent = ROOT.parent
    for folder in [parent / "JPG素材", parent / "JPG材料"]:
        if folder.exists():
            jpgs = sorted(folder.glob("*.jpg"))
            if jpgs:
                return jpgs[0]
    raise FileNotFoundError("未找到示例 JPG，请手动传入路径")


def main():
    img = Path(sys.argv[1]) if len(sys.argv) > 1 else find_sample()
    print("使用图像:", img)
    state = run_full_demo(str(img))
    summary = {
        "session_id": state["session_id"],
        "stage": state["stage"],
        "experience_targets": state["experience_targets"],
        "target_metrics": state["confirmed_target_metrics"],
        "modification_plan": (state.get("modification_plan") or {}).get("draft_text", "")[:120],
        "output_image": (state.get("generation") or {}).get("output_image_path"),
        "mock": (state.get("generation") or {}).get("mock"),
        "quality_passed": (state.get("quality_report") or {}).get("passed"),
        "memory_id": state.get("memory_id"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
