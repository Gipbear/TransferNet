"""统一 dump CLI：producer 产 ScoreBundle → 存兼容 .pt。"""
from __future__ import annotations

import argparse
import os
import time
from copy import copy
from pathlib import Path
from typing import Any

import torch

from kgqa.backbone import make_score_producer
from kgqa.runtime import add_runtime_arguments, configure_runtime, emit_event, update_progress


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="kgqa 统一得分 dump")
    p.add_argument("--dataset", required=True, help="数据集：webqsp | metaqa | cwq")
    p.add_argument("--backbone", default="transfernet", choices=["transfernet", "rearev"],
                   help="基础检索模型；ReaRev 当前不支持生成 score 缓存")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--input_dir", required=True)
    p.add_argument("--qa_file", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--output", required=True)
    p.add_argument("--topk", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--bert_name", default=None)
    p.add_argument("--per_hop_limit", type=int, default=0,
                   help="MetaQA 每跳保留前 N 条（分层小子集），0=全量")
    p.add_argument("--limit", type=int, default=0,
                   help="CWQ 取前 N 条非空子图样本（小子集），0=全量")
    p.add_argument("--rev", action="store_true",
                   help="CWQ 补反向关系（关系词表翻倍）；须与 ckpt 的训练设置一致。"
                        "WebQSP 的反向边已烘焙进 relations.dict，无需该开关")
    add_runtime_arguments(p)
    return p


def _bundle_to_cache(bundle) -> dict:
    meta = bundle.meta
    return {
        "version": 1,
        "meta": {"dataset": meta.dataset, "split": meta.split,
                 "num_samples": meta.num_samples, "topk_entities": meta.topk_entities,
                 "input_dir": meta.input_dir, "qa_file": meta.qa_file,
                 "id2ent": meta.id2ent, "id2rel": meta.id2rel},
        "samples": [{
            "question": s.question, "topic_ids": s.topic_ids, "gold_ids": s.gold_ids,
            "hop_attn": s.hop_attn, "rel_probs": s.rel_probs,
            "ent_indices": s.ent_indices, "ent_scores": s.ent_scores,
            "e_score_indices": s.e_score_indices, "e_score_values": s.e_score_values,
            **({"hop": s.hop} if s.hop is not None else {}),
            **({"triples": s.triples} if s.triples is not None else {}),
        } for s in bundle.samples],
    }


def truncate_score_cache(cache: dict[str, Any], topk: int) -> dict[str, Any]:
    """从较大 top-k 缓存精确裁剪出较小的得分缓存。

    各 ScoreProducer 在保存前已按分数降序执行 ``topk``，因此无并列边界时保留
    前 N 项与直接以 N 运行前向推理一致。clone 用于避免小缓存仍引用大缓存的底层存储。
    """
    if topk <= 0:
        raise ValueError("topk 必须为正整数")
    meta = cache.get("meta")
    samples = cache.get("samples")
    if not isinstance(meta, dict) or not isinstance(samples, list):
        raise ValueError("score 缓存格式不完整，无法裁剪")
    source_topk = meta.get("topk_entities")
    if not isinstance(source_topk, int) or source_topk < topk:
        raise ValueError(f"源缓存 top-k={source_topk} 小于目标 top-k={topk}")

    truncated_samples: list[dict[str, Any]] = []
    for sample in samples:
        item = copy(sample)
        item["ent_indices"] = [values[:topk].clone() for values in sample["ent_indices"]]
        item["ent_scores"] = [values[:topk].clone() for values in sample["ent_scores"]]
        item["e_score_indices"] = sample["e_score_indices"][:topk].clone()
        item["e_score_values"] = sample["e_score_values"][:topk].clone()
        truncated_samples.append(item)
    return {
        **cache,
        "meta": {**meta, "topk_entities": topk},
        "samples": truncated_samples,
    }


def materialize_truncated_score_cache(source: str | Path, output: str | Path, topk: int) -> None:
    """加载一次已有大缓存并写出指定 top-k 的独立缓存文件。"""
    source_path = Path(source)
    output_path = Path(output)
    cache = torch.load(source_path, map_location="cpu", weights_only=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(truncate_score_cache(cache, topk), output_path)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.backbone != "transfernet":
        raise SystemExit("ReaRev 当前只支持消费既有离线 score 缓存，不能 dump_scores")
    run_dir = configure_runtime(
        args, command="生成检索得分缓存",
        fallback_run_dir=os.path.dirname(os.path.abspath(args.output)),
        manifest={"dataset": args.dataset, "backbone": args.backbone, "split": args.split,
                  "topk": args.topk, "output": os.path.abspath(args.output)},
    )
    try:
        producer = make_score_producer(
            args.dataset,
            bert_name=args.bert_name,
            per_hop_limit=args.per_hop_limit,
            limit=args.limit,
            rev=args.rev,
        )
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc
    producer.load_checkpoint(args.ckpt)
    interval = max(args.progress_interval, 1)
    last_completed = 0

    def report_progress(completed: int, total: int) -> None:
        nonlocal last_completed
        if completed - last_completed >= interval or completed == total:
            update_progress(run_dir, completed=completed, total=total, phase="生成得分缓存")
            last_completed = completed

    started = time.perf_counter()
    bundle = producer.produce(args.input_dir, args.qa_file, split=args.split,
                              batch_size=args.batch_size, topk=args.topk,
                              show_progress=not args.no_progress,
                              progress_callback=report_progress)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(_bundle_to_cache(bundle), args.output)
    update_progress(run_dir, completed=len(bundle.samples), total=len(bundle.samples),
                    status="completed", phase="生成得分缓存")
    emit_event(run_dir, "phase_end", phase="生成得分缓存", samples=len(bundle.samples))
    elapsed = time.perf_counter() - started
    speed = len(bundle.samples) / elapsed if elapsed else 0.0
    print(f"[INFO] dump 完成 {len(bundle.samples)} 条，耗时 {elapsed:.1f} 秒，"
          f"{speed:.2f} 题/s → {args.output}", flush=True)


if __name__ == "__main__":
    main()
