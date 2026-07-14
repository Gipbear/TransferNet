"""统一检索 CLI。"""
from __future__ import annotations

import argparse
import json
import os

from tqdm import tqdm

from kgqa.retrieve.datasets.registry import get_adapter
from kgqa.backbone import make_score_producer
from kgqa.runtime import add_runtime_arguments, configure_runtime, emit_event, update_progress


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="kgqa 统一路径检索")
    p.add_argument("--dataset", required=True, help="数据集：webqsp | metaqa | cwq")
    p.add_argument("--backbone", default="transfernet", choices=["transfernet", "rearev"],
                   help="基础检索模型；ReaRev 当前仅支持 WebQSP 离线缓存")
    p.add_argument("--backend", choices=["offline", "online"], default="offline")
    p.add_argument("--cache", default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--input_dir", required=True)
    p.add_argument("--qa_file", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--beam_size", type=int, default=50)
    p.add_argument("--lambda_val", type=float, default=0.2)
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--eta", type=float, default=1.0,
                   help="终点实体分数融合权重 η（论文符号）")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output", default=None, help="逐样本 JSONL 输出路径")
    add_runtime_arguments(p)
    return p


def _make_producer(dataset: str, backbone: str = "transfernet"):
    if backbone != "transfernet":
        raise SystemExit("ReaRev 当前只支持离线 score 缓存检索，不能 online 检索")
    try:
        return make_score_producer(dataset)
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc


def _adapter_name(dataset: str, backbone: str) -> str:
    """将 dataset/backbone 的正交接口映射到保留的适配器兼容别名。"""
    if dataset == "webqsp-rearev":
        return dataset
    if backbone == "rearev":
        if dataset != "webqsp":
            raise SystemExit("ReaRev 当前只支持 WebQSP")
        return "webqsp-rearev"
    return dataset


def build_backend(args):
    adapter_name = _adapter_name(args.dataset, args.backbone)
    adapter = get_adapter(adapter_name, input_dir=args.input_dir)
    if args.backend == "offline":
        from kgqa.retrieve.backends.offline import OfflineBackend
        if not args.cache:
            raise SystemExit("--backend offline 需要 --cache")
        return OfflineBackend(adapter, cache_path=args.cache)
    from kgqa.retrieve.backends.online import OnlineBackend
    if not (args.ckpt and args.qa_file):
        raise SystemExit("--backend online 需要 --ckpt 和 --qa_file")
    backend = OnlineBackend(adapter, _make_producer(args.dataset, args.backbone), ckpt_path=args.ckpt,
                            input_dir=args.input_dir, qa_file=args.qa_file,
                            split=args.split)
    return backend


def run_retrieval(args):
    fallback = os.path.dirname(os.path.abspath(args.output)) if args.output else None
    run_dir = configure_runtime(
        args, command="路径检索",
        fallback_run_dir=fallback,
        manifest={"dataset": args.dataset, "backbone": args.backbone, "backend": args.backend,
                  "cache": args.cache, "output": args.output},
    )
    backend = build_backend(args)
    params = dict(beam_size=args.beam_size, lambda_val=args.lambda_val,
                  threshold=args.threshold, eta=args.eta)
    samples = backend.bundle.samples[:args.limit] if args.limit else backend.bundle.samples
    total = len(samples)
    results = []
    interval = max(args.progress_interval, 1)
    with tqdm(total=total, desc="路径检索", unit="题", dynamic_ncols=True,
              disable=args.no_progress) as progress:
        for sample_index in range(total):
            results.append(backend.retrieve(sample_index, **params))
            progress.update(1)
            if progress.n % interval == 0 or progress.n == total:
                update_progress(run_dir, completed=progress.n, total=total, phase="路径检索")
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps({
                    "sample_index": r.sample_index, "question": r.question,
                    "topics": r.topics, "hop": r.hop, "golden": r.golden,
                    "mmr_reason_paths": r.paths, "prediction": r.prediction,
                }, ensure_ascii=False) + "\n")
    update_progress(run_dir, completed=len(results), total=len(results), status="completed", phase="路径检索")
    emit_event(run_dir, "phase_end", phase="路径检索", samples=len(results))
    return backend, results


def main(argv=None):
    args = build_parser().parse_args(argv)
    _backend, results = run_retrieval(args)
    print(f"[INFO] 检索完成 {len(results)} 条", flush=True)


if __name__ == "__main__":
    main()
