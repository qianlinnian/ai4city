"""使用仓库 p1 种子案例演示 Task 2 → Task 3；无需 API Key。"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.cartographer_agent import CartographerAgent
from agents.translator_agent import TranslatorAgent
from knowledge_base import kb


def main() -> None:
    seed = next(record for record in kb.list_experience_cases() if record["id"] == "seed-local-p1")
    image_path = ROOT / "data" / "p1.jpg"
    scene_context = "社区绿地；午后；中等人流"

    translation = TranslatorAgent().run(
        original_image_path=str(image_path),
        experience_baseline=seed["experience_baseline"],
        experience_targets=seed["experience_targets"],
        baseline_metrics=seed["baseline_metrics"],
        scene_context=scene_context,
    )

    layout = CartographerAgent().run(
        original_image_path=str(image_path),
        baseline_metrics=seed["baseline_metrics"],
        target_metrics=translation.target_metrics.as_dict(),
        experience_baseline=seed["experience_baseline"],
        experience_targets=seed["experience_targets"],
        scene_context=scene_context,
        expert_advice="保留主要道路与建筑主体，优先改善停留节点",
        language="zh",
    )

    print(
        json.dumps(
            {
                "task2": translation.model_dump(),
                "task3": layout.model_dump(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
