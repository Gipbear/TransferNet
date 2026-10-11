"""TransferNet 候选答案最短路径后处理基线 CLI。"""
from __future__ import annotations

import argparse
import os
import time
from types import SimpleNamespace

from tqdm import tqdm

from kgqa.retrieve.cli.eval import evaluate_results, write_results
from kgqa.retrieve.cli.retrieve import _adapter_name
from kgqa.retrieve.datasets.registry import get_adapter
from kgqa.retrieve.shortest_path import ShortestPathParams, retrieve_shortest_paths_one
from kgqa.runtime import add_runtime_arguments, configure_runtime, emit_event, file_fingerprint, update_progress


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="基于骨干候选答案的最短路径后处理基线")
    parser.add_argument("--dataset", required=True, help="数据集：webqsp | metaqa | cwq")
    parser.add_argument("--backbone", default="transfernet", choices=["transfernet", "rearev"],
                        help="基础检索模型；本基线当前使用离线得分缓存")
    parser.add_argument("--cache", required=True, help="已有离线 score 缓存")
    parser.add_argument("--input_dir", required=True, help="数据集输入目录，用于构建知识图谱邻接表")
    parser.add_argument("--candidate_topk", type=int, default=20, help="最终实体分数的候选答案数量")
    parser.add_argument("--max_paths_per_pair", type=int, default=20, help="每个主题实体与候选答案对保留的最短路径数")
    parser.add_argument("--path_budget", type=int, default=20, help="每题最终保留的路径数量")
    parser.add_argument("--drop_loopback", action=argparse.BooleanOptionalAction, default=True,
                        help="是否剔除终点等于主题实体的路径，默认启用")
    parser.add_argument("--limit", type=int, default=0, help="仅运行前 N 条，0 表示全部")
    parser.add_argument("--output", required=True, help="逐样本 JSONL 输出路径")
    parser.add_argument("--summary", required=True, help="汇总 JSON 输出路径")
    add_runtime_arguments(parser)
    return parser


def run_shortest_path(args: argparse.Namespace):
    params = ShortestPathParams(
        candidate_topk=args.candidate_topk,
        max_paths_per_pair=args.max_paths_per_pair,
        path_budget=args.path_budget,
        drop_loopback=args.drop_loopback,
    )
    adapter = get_adapter(_adapter_name(args.dataset, args.backbone), input_dir=args.input_dir)
    bundle = adapter.score_loader().load(args.cache)
    samples = bundle.samples[:args.limit] if args.limit else bundle.samples
    edge_source = adapter.kg_edge_source(samples[0]) if samples else None
    run_dir = configure_runtime(
        args,
        command="候选答案最短路径后处理",
        fallback_run_dir=os.path.dirname(os.path.abspath(args.output)),
        manifest={
            "dataset": args.dataset,
            "backbone": args.backbone,
            "method": "shortest_path_postprocess",
            "cache": file_fingerprint(args.cache),
            "candidate_topk": params.candidate_topk,
            "max_paths_per_pair": params.max_paths_per_pair,
            "path_budget": params.path_budget,
            "max_hop_source": "available_steps",
            "drop_loopback": params.drop_loopback,
            "edge_source": type(edge_source).__name__ if edge_source else None,
            "path_sort_version": 1,
            "output": os.path.abspath(args.output),
            "summary": os.path.abspath(args.summary),
        },
    )
    started = time.perf_counter()
    results = []
    interval = max(args.progress_interval, 1)
    with tqdm(total=len(samples), desc="最短路径后处理", unit="题", dynamic_ncols=True,
              disable=args.no_progress) as progress:
        for sample in samples:
            results.append(retrieve_shortest_paths_one(
                sample,
                adapter.kg_edge_source(sample),
                bundle.meta.id2ent,
                bundle.meta.id2rel,
                params=params,
            ))
            progress.update(1)
            if progress.n % interval == 0 or progress.n == len(samples):
                update_progress(run_dir, completed=progress.n, total=len(samples), phase="最短路径后处理")
    write_results(results, args.output)
    backend = SimpleNamespace(adapter=adapter, bundle=bundle)
    summary = evaluate_results(backend, results, args.summary)
    update_progress(run_dir, completed=len(results), total=len(samples), status="completed", phase="最短路径后处理")
    emit_event(run_dir, "phase_end", phase="最短路径后处理", samples=len(results))
    elapsed = time.perf_counter() - started
    speed = len(results) / elapsed if elapsed else 0.0
    print(f"[INFO] 最短路径后处理完成 {len(results)} 条，耗时 {elapsed:.1f} 秒，{speed:.2f} 题/s", flush=True)
    return results, summary


def main(argv: list[str] | None = None) -> int:
    run_shortest_path(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
