"""使用 DeepSeek 将规范 PDF 整理为可追溯的结构化知识草稿。

默认仅打印执行计划，不产生 API 费用。必须显式传入 ``--execute`` 才会调用
DeepSeek；原 PDF 始终只读，草稿写入项目 outputs 目录。
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import KNOWLEDGE_DRAFT_DIR, KNOWLEDGE_SOURCE_DIR  # noqa: E402
from knowledge_base.knowledge_curator import (  # noqa: E402
    DeepSeekKnowledgeClient,
    batch_output_path,
    build_pdf_batches,
    discover_knowledge_pdfs,
    payload_status,
    process_batch,
    revalidate_draft_file,
    write_batch_draft,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DeepSeek 离线知识整理器（默认 dry-run，不调用 API）"
    )
    parser.add_argument("--source-dir", type=Path, default=KNOWLEDGE_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=KNOWLEDGE_DRAFT_DIR)
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="只处理文件名中包含该文字的 PDF；可重复传入",
    )
    parser.add_argument("--max-chars", type=int, default=12000)
    parser.add_argument("--max-pages", type=int, default=4)
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="跳过结束页仍小于该页码的前置批次，例如正文从第 10 页开始",
    )
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=None,
        help="仅处理前 N 个批次，适合先小规模付费试跑",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="真实调用 DeepSeek；不传时只打印计划",
    )
    parser.add_argument(
        "--revalidate-existing",
        action="store_true",
        help="仅重校验已有草稿，不调用 DeepSeek API",
    )
    parser.add_argument(
        "--auto-pro",
        action="store_true",
        help="Flash 结果需复核时，再调用 Pro 重新整理该批次",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="覆盖同一源文件指纹下已经生成的批次草稿",
    )
    parser.add_argument(
        "--retry-invalid",
        action="store_true",
        help="只重做已有但含校验错误的批次；正常 empty 和有效草稿仍会跳过",
    )
    parser.add_argument(
        "--retry-empty",
        action="store_true",
        help="重做已有且状态为 empty 的批次；用于抽取规则或 Prompt 更新后复查",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并发 API 请求数；默认 1，批量任务建议先使用 16～32，上限 64",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="将任务确定性分成多少份，供多个终端并行",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="当前终端处理的分片编号，范围 0 到 shard-count-1",
    )
    return parser


def select_shard(items: list[Any], shard_count: int, shard_index: int) -> list[Any]:
    if shard_count < 1:
        raise ValueError("--shard-count 必须大于 0")
    if not 0 <= shard_index < shard_count:
        raise ValueError("--shard-index 必须位于 0 到 shard-count-1")
    return [item for index, item in enumerate(items) if index % shard_count == shard_index]


def draft_requires_api_retry(path: str | Path) -> bool:
    """判断已有草稿是否因结构/证据错误而需要重新请求模型。"""

    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return True
    return bool(document.get("validation_errors"))


def draft_is_empty(path: str | Path) -> bool:
    """判断已有草稿是否是模型明确返回的空知识批次。"""

    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return document.get("review_status") == "empty"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.force and (args.retry_invalid or args.retry_empty):
        raise ValueError("--force 不能与 --retry-invalid/--retry-empty 同时使用")
    if not 1 <= args.workers <= 64:
        raise ValueError("--workers 必须位于 1 到 64；批量任务建议先使用 16～32")
    # 即使是重校验模式也校验分片参数，避免命令拼写错误被静默忽略。
    select_shard([], args.shard_count, args.shard_index)
    if args.revalidate_existing:
        if args.execute:
            raise ValueError("--revalidate-existing 不得与 --execute 同时使用")
        drafts = sorted(Path(args.output_dir).resolve().rglob("batch-*.json"))
        if args.include:
            drafts = [
                path
                for path in drafts
                if any(
                    value.casefold() in path.parent.name.casefold()
                    for value in args.include
                )
            ]
        if args.limit_batches is not None:
            if args.limit_batches < 1:
                raise ValueError("--limit-batches 必须大于 0")
            drafts = drafts[: args.limit_batches]
        drafts = select_shard(drafts, args.shard_count, args.shard_index)
        if not drafts:
            print("没有找到可重校验的草稿。")
            return 1
        failed = 0
        for draft in drafts:
            try:
                payload = revalidate_draft_file(draft)
                print(
                    f"{draft}: records={len(payload.records)} "
                    f"status={payload_status(payload)}"
                )
            except Exception as exc:
                failed += 1
                print(f"{draft}: 重校验失败: {type(exc).__name__}: {exc}")
        print(f"重校验完成：成功 {len(drafts) - failed}，失败 {failed}；未调用 API。")
        return 1 if failed else 0

    pdfs = discover_knowledge_pdfs(args.source_dir)
    if args.include:
        pdfs = [
            path
            for path in pdfs
            if any(value.casefold() in path.name.casefold() for value in args.include)
        ]
    if not pdfs:
        print("没有找到符合条件的 PDF。")
        return 1
    if args.start_page < 1:
        raise ValueError("--start-page 必须大于 0")

    planned: list[tuple[object, Path]] = []
    for pdf in pdfs:
        batches = build_pdf_batches(
            pdf, max_chars=args.max_chars, max_pages=args.max_pages
        )
        print(f"{pdf.name}: {len(batches)} 个批次")
        for batch in batches:
            if max(batch.page_numbers) < args.start_page:
                continue
            output_path = batch_output_path(args.output_dir, batch)
            if output_path.is_file() and not args.force:
                retry_invalid = args.retry_invalid and draft_requires_api_retry(
                    output_path
                )
                retry_empty = args.retry_empty and draft_is_empty(output_path)
                if not (retry_invalid or retry_empty):
                    continue
            planned.append((batch, output_path))

    planned = select_shard(planned, args.shard_count, args.shard_index)
    if args.limit_batches is not None:
        if args.limit_batches < 1:
            raise ValueError("--limit-batches 必须大于 0")
        planned = planned[: args.limit_batches]

    print(f"待处理批次: {len(planned)}")
    print(f"草稿目录: {Path(args.output_dir).resolve()}")
    print(f"自动 Pro 复核: {'开启' if args.auto_pro else '关闭'}")
    print(f"并发请求数: {args.workers}")
    print(f"任务分片: {args.shard_index}/{args.shard_count}")
    if not args.execute:
        print("DRY-RUN：未调用任何 API。确认后添加 --execute。")
        return 0

    if not planned:
        print("没有待处理批次；未调用 API。")
        return 0

    thread_state = threading.local()

    def client_for_worker() -> DeepSeekKnowledgeClient:
        client = getattr(thread_state, "client", None)
        if client is None:
            client = DeepSeekKnowledgeClient()
            thread_state.client = client
        return client

    def execute_one(index: int, batch: Any, output_path: Path):
        model, tier, payload = process_batch(
            client_for_worker(), batch, auto_pro=args.auto_pro
        )
        write_batch_draft(
            output_path,
            batch=batch,
            model=model,
            tier=tier,
            payload=payload,
        )
        return index, batch, output_path, payload

    succeeded = 0
    failed = 0
    futures = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, (batch, output_path) in enumerate(planned, start=1):
            print(
                f"[{index}/{len(planned)}] 提交 {batch.source.name} "
                f"{batch.batch_id} pages={batch.page_numbers}"
            )
            future = executor.submit(execute_one, index, batch, output_path)
            futures[future] = (index, batch, output_path)

        for future in as_completed(futures):
            index, batch, output_path = futures[future]
            try:
                _, _, _, payload = future.result()
            except Exception as exc:
                failed += 1
                print(
                    f"[{index}/{len(planned)}] {batch.source.name} "
                    f"{batch.batch_id} 失败: {type(exc).__name__}: {exc}"
                )
                continue
            succeeded += 1
            print(
                f"[{index}/{len(planned)}] 完成 -> {output_path} "
                f"records={len(payload.records)} "
                f"status={payload_status(payload)}"
            )

    print(f"完成：成功 {succeeded}，失败 {failed}。原始 PDF 未修改。")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
