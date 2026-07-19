"""使用 DeepSeek 将 prompt.docx 整理为 Task 3 场景案例 RAG JSON。

默认 dry-run，不调用 API。只有显式传入 ``--execute`` 才会产生模型费用。
Flash 结果未通过门禁时默认自动调用 Pro 修复；原 DOCX 始终只读。
"""

from __future__ import annotations

import argparse
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import KNOWLEDGE_DRAFT_DIR, RAG_PUBLISHED_KNOWLEDGE_DIR  # noqa: E402
from knowledge_base.knowledge_curator import DeepSeekKnowledgeClient  # noqa: E402
from knowledge_base.scene_prompt_curator import (  # noqa: E402
    curate_scene_case,
    draft_path,
    file_sha256,
    parse_scene_prompt_docx,
    publish_scene_prompt_cases,
    revalidate_scene_prompt_draft,
    reusable_draft,
    write_json_atomic,
)


DEFAULT_DRAFT_DIR = KNOWLEDGE_DRAFT_DIR / "scene_prompt_cases"
DEFAULT_PUBLISHED_PATH = RAG_PUBLISHED_KNOWLEDGE_DIR / "scene_prompt_examples.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DeepSeek 场景 Prompt 案例整理器（默认 dry-run，不调用 API）"
    )
    parser.add_argument("--source", type=Path, default=ROOT / "prompt.docx")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DRAFT_DIR)
    parser.add_argument("--published-path", type=Path, default=DEFAULT_PUBLISHED_PATH)
    parser.add_argument(
        "--scene-type",
        choices=["blue_green", "commercial_office", "community"],
        action="append",
        default=[],
        help="只处理指定场景；可重复传入",
    )
    parser.add_argument(
        "--limit-cases",
        type=int,
        default=None,
        help="只处理前 N 个待处理案例，适合小规模付费验证",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="并发 API 请求数，范围 1～16，默认 4",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="真实调用 DeepSeek；不传时只解析文档并打印计划",
    )
    parser.add_argument(
        "--no-auto-pro",
        action="store_true",
        help="关闭默认的 Flash 失败后 Pro 自动修复",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="重新处理已通过程序门禁的同源案例（会再次产生费用）",
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="即使 30 个案例全部通过也不写正式 RAG JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.workers <= 16:
        raise ValueError("--workers 必须位于 1 到 16")
    if args.limit_cases is not None and args.limit_cases < 1:
        raise ValueError("--limit-cases 必须大于 0")

    source = args.source.resolve()
    cases = parse_scene_prompt_docx(source)
    source_fingerprint = file_sha256(source)
    selected = [
        case for case in cases if not args.scene_type or case.scene_type in args.scene_type
    ]
    counts = {
        scene_type: sum(case.scene_type == scene_type for case in cases)
        for scene_type in ("blue_green", "commercial_office", "community")
    }
    print(f"文档: {source}")
    print(f"解析案例: {len(cases)}，分类={counts}")
    print(f"原文 SHA256: {source_fingerprint[:12]}…")

    planned = []
    reused = 0
    for case in selected:
        output = draft_path(args.output_dir, case)
        if output.is_file() and not args.force:
            revalidate_scene_prompt_draft(
                output,
                case,
                source_name=source.name,
                source_fingerprint=source_fingerprint,
            )
        if not args.force and reusable_draft(
            output, case, source_fingerprint=source_fingerprint
        ):
            reused += 1
            continue
        planned.append((case, output))
    if args.limit_cases is not None:
        planned = planned[: args.limit_cases]

    print(f"已通过且可复用: {reused}")
    print(f"待处理案例: {len(planned)}")
    print(f"草稿目录: {args.output_dir.resolve()}")
    print(f"自动 Pro 修复: {'关闭' if args.no_auto_pro else '开启'}")
    print(f"并发请求数: {args.workers}")
    if not args.execute:
        for case, _ in planned[:10]:
            print(
                f"- {case.knowledge_id}: {case.image_id}, "
                f"paragraphs={case.paragraphs[0].paragraph_id}..{case.paragraphs[-1].paragraph_id}"
            )
        if len(planned) > 10:
            print(f"- …其余 {len(planned) - 10} 个案例")
        print("DRY-RUN：未调用任何 API。确认后添加 --execute。")
        return 0

    thread_state = threading.local()

    def client_for_worker() -> DeepSeekKnowledgeClient:
        client = getattr(thread_state, "client", None)
        if client is None:
            client = DeepSeekKnowledgeClient()
            thread_state.client = client
        return client

    def execute_one(index: int, case, output: Path):
        draft = curate_scene_case(
            client_for_worker(),
            case,
            source_name=source.name,
            source_fingerprint=source_fingerprint,
            auto_pro=not args.no_auto_pro,
        )
        write_json_atomic(output, draft)
        return index, case, output, draft

    succeeded = 0
    held = 0
    failed = 0
    futures = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, (case, output) in enumerate(planned, start=1):
            print(
                f"[{index}/{len(planned)}] 提交 {case.knowledge_id} "
                f"{case.paragraphs[0].paragraph_id}..{case.paragraphs[-1].paragraph_id}"
            )
            future = executor.submit(execute_one, index, case, output)
            futures[future] = (index, case)
        for future in as_completed(futures):
            index, case = futures[future]
            try:
                _, _, output, draft = future.result()
            except Exception as exc:
                failed += 1
                print(
                    f"[{index}/{len(planned)}] {case.knowledge_id} 失败: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            status = draft["review_status"]
            if status == "program_validated":
                succeeded += 1
            else:
                held += 1
            tiers = "→".join(attempt["tier"] for attempt in draft["attempts"])
            print(
                f"[{index}/{len(planned)}] 完成 -> {output} "
                f"status={status} model={tiers}"
            )

    print(f"处理完成：通过 {succeeded}，留置 {held}，失败 {failed}。原 DOCX 未修改。")
    if not args.no_publish:
        published, missing = publish_scene_prompt_cases(
            cases,
            draft_dir=args.output_dir,
            output_path=args.published_path,
            source_name=source.name,
            source_fingerprint=source_fingerprint,
        )
        if published is None:
            print(
                f"正式库暂未更新：仍有 {len(missing)} 个案例未通过程序门禁。"
            )
        else:
            print(f"正式 RAG 案例库: {published}（30 条）")
    return 1 if failed or held else 0


if __name__ == "__main__":
    raise SystemExit(main())
