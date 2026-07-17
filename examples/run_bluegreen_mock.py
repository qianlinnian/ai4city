"""用蓝绿-01/02/03临时数据运行 Task 2 与 Task 3。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.cartographer_agent import CartographerAgent  # noqa: E402
from agents.translator_agent import TranslatorAgent  # noqa: E402


DATA_PATH = ROOT / "examples" / "bluegreen_01_03_mock_data.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="运行蓝绿-01/02/03临时联调数据")
    parser.add_argument(
        "--scene",
        choices=["蓝绿-01", "蓝绿-02", "蓝绿-03", "all"],
        default="all",
    )
    parser.add_argument("--language", choices=["zh", "en"], default="zh")
    parser.add_argument("--output", type=Path, help="可选：保存完整输出JSON")
    args = parser.parse_args()

    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    records = payload["records"]
    if args.scene != "all":
        records = [record for record in records if record["scene_id"] == args.scene]

    translator = TranslatorAgent()
    cartographer = CartographerAgent()
    outputs = []
    for record in records:
        translation = translator.run(
            experience_baseline=record["experience_baseline"],
            experience_targets=record["experience_targets"],
            baseline_metrics=record["baseline_metrics"],
            scene_context=record["scene_context_text"],
            original_image_path=record["original_image_path"],
        )
        plan = cartographer.run(
            baseline_metrics=translation.baseline_metrics.as_dict(),
            target_metrics=translation.target_metrics.as_dict(),
            experience_baseline=record["experience_baseline"],
            experience_targets=record["experience_targets"],
            scene_context=record["scene_context_text"],
            original_image_path=record["original_image_path"],
            expert_advice="临时联调数据：优先增加绿化和小尺度可见水体，保持道路与建筑结构不变",
            language=args.language,
        )
        outputs.append(
            {
                "scene_id": record["scene_id"],
                "data_status": payload["data_status"],
                "input": record,
                "task2_translation": translation.model_dump(),
                "task3_modification_plan": plan.model_dump(),
            }
        )

    text = json.dumps(outputs, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"已写入: {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
