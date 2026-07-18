"""发布通过质量门禁的知识；默认 dry-run，不调用任何模型 API。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import KNOWLEDGE_DRAFT_DIR, RAG_PUBLISHED_KNOWLEDGE_DIR  # noqa: E402
from knowledge_base.knowledge_publisher import (  # noqa: E402
    PublishPolicy,
    collect_publication_records,
    write_publication,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="发布正式 RAG 知识（默认 dry-run）")
    parser.add_argument("--draft-dir", type=Path, default=KNOWLEDGE_DRAFT_DIR)
    parser.add_argument("--output-dir", type=Path, default=RAG_PUBLISHED_KNOWLEDGE_DIR)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--min-fuzzy-score", type=float, default=0.85)
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.min_confidence <= 1.0:
        raise ValueError("--min-confidence 必须在 0～1")
    if not 0.0 <= args.min_fuzzy_score <= 1.0:
        raise ValueError("--min-fuzzy-score 必须在 0～1")
    policy = PublishPolicy(args.min_confidence, args.min_fuzzy_score)
    published, held = collect_publication_records(
        args.draft_dir,
        include=args.include,
        policy=policy,
    )
    print(f"可发布记录: {len(published)}")
    print(f"暂缓记录: {len(held)}")
    print(f"正式目录: {Path(args.output_dir).resolve()}")
    if not args.execute:
        print("DRY-RUN：未写入正式知识库，未调用任何 API。")
        return 0
    paths = write_publication(published, output_dir=args.output_dir)
    report_path = Path(args.output_dir).resolve().parent / "last_publish_report.json"
    report_path.write_text(
        json.dumps(
            {
                "published_count": len(published),
                "held_count": len(held),
                "output_files": [str(path) for path in paths],
                "held_records": held,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"发布完成：{len(paths)} 个知识文件；报告：{report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
