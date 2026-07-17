"""Offline-safe command-line demo for the Excel-driven pipeline.

Usage::

    python run_demo.py pano.jpg --baseline-json baseline.json

``baseline.json`` is the seven-key row already selected from the project data
table.  The demo always uses the local mock generator and never calls a paid
service.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.generation_backend import generate
from pipeline.orchestrator import run_full_demo


def _mock_generator(image_path: str | Path, prompt: str):
    return generate(image_path, prompt, backend="mock")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用外部七项基线指标运行本地 MOCK 全流程演示"
    )
    parser.add_argument("image", type=Path, help="原始全景图路径")
    parser.add_argument(
        "--baseline-json",
        type=Path,
        required=True,
        help="包含七项英文内部字段的 JSON 文件",
    )
    args = parser.parse_args()

    try:
        baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        parser.error(f"基线 JSON 格式错误：{exc.msg}")
    if not isinstance(baseline, dict):
        parser.error("基线 JSON 顶层必须是对象")

    state = run_full_demo(
        str(args.image),
        baseline,
        generator=_mock_generator,
    )
    summary = {
        "session_id": state["session_id"],
        "stage": state["stage"],
        "experience_targets": state["experience_targets"],
        "target_metrics": state["confirmed_target_metrics"],
        "modification_plan": (
            state.get("modification_plan") or {}
        ).get("draft_text", "")[:120],
        "output_image": (state.get("generation") or {}).get("output_image_path"),
        "mock": (state.get("generation") or {}).get("mock"),
        "quality_report": state.get("quality_report"),
        "memory_id": state.get("memory_id"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
