"""统一 dump CLI：producer 产 ScoreBundle → 存兼容 .pt。"""
from __future__ import annotations

import argparse
import os

import torch


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="kgqa 统一得分 dump")
    p.add_argument("--dataset", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--input_dir", required=True)
    p.add_argument("--qa_file", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--output", required=True)
    p.add_argument("--topk", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--per_hop_limit", type=int, default=0,
                   help="MetaQA 每跳保留前 N 条（分层小子集），0=全量")
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


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.dataset == "webqsp":
        from kgqa.models.webqsp import WebQSPScoreProducer
        producer = WebQSPScoreProducer()
    elif args.dataset == "metaqa":
        from kgqa.models.metaqa import MetaQAScoreProducer
        producer = MetaQAScoreProducer(per_hop_limit=args.per_hop_limit)
    else:
        raise SystemExit(f"未支持的 dump 数据集: {args.dataset}")
    producer.load_checkpoint(args.ckpt)
    bundle = producer.produce(args.input_dir, args.qa_file, split=args.split,
                              batch_size=args.batch_size, topk=args.topk)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(_bundle_to_cache(bundle), args.output)
    print(f"[INFO] dump 完成 {len(bundle.samples)} 条 → {args.output}", flush=True)


if __name__ == "__main__":
    main()
