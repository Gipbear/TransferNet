"""统一检索 CLI。"""
from __future__ import annotations

import argparse
import json
import os

from kgqa.datasets.registry import get_adapter
from kgqa.backbone import make_score_producer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="kgqa 统一路径检索")
    p.add_argument("--dataset", required=True)
    p.add_argument("--backend", choices=["offline", "online"], default="offline")
    p.add_argument("--cache", default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--input_dir", required=True)
    p.add_argument("--qa_file", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--beam_size", type=int, default=50)
    p.add_argument("--lambda_val", type=float, default=0.2)
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--alpha_final", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output", default=None, help="逐样本 jsonl")
    return p


def _make_producer(dataset: str):
    try:
        return make_score_producer(dataset)
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc


def build_backend(args):
    adapter = get_adapter(args.dataset, input_dir=args.input_dir)
    if args.backend == "offline":
        from kgqa.retrieve.backends.offline import OfflineBackend
        if not args.cache:
            raise SystemExit("--backend offline 需要 --cache")
        return OfflineBackend(adapter, cache_path=args.cache)
    from kgqa.retrieve.backends.online import OnlineBackend
    if not (args.ckpt and args.qa_file):
        raise SystemExit("--backend online 需要 --ckpt 和 --qa_file")
    backend = OnlineBackend(adapter, _make_producer(args.dataset), ckpt_path=args.ckpt,
                            input_dir=args.input_dir, qa_file=args.qa_file,
                            split=args.split)
    return backend


def run_retrieval(args):
    backend = build_backend(args)
    params = dict(beam_size=args.beam_size, lambda_val=args.lambda_val,
                  threshold=args.threshold, alpha_final=args.alpha_final)
    results = backend.retrieve_all(limit=args.limit, **params)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps({
                    "sample_index": r.sample_index, "question": r.question,
                    "topics": r.topics, "hop": r.hop, "golden": r.golden,
                    "mmr_reason_paths": r.paths, "prediction": r.prediction,
                }, ensure_ascii=False) + "\n")
    return backend, results


def main(argv=None):
    args = build_parser().parse_args(argv)
    _backend, results = run_retrieval(args)
    print(f"[INFO] 检索完成 {len(results)} 条", flush=True)


if __name__ == "__main__":
    main()
