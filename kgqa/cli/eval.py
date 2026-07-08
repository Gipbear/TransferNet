"""统一评测 CLI：检索 + 答案级/路径级指标。"""
from __future__ import annotations

import argparse
import json
import os

from kgqa.cli.retrieve import build_parser as _retrieve_parser, run_retrieval
from kgqa.eval.answer_eval import answer_record, answer_summary
from kgqa.eval.path_eval import path_summary


def build_parser() -> argparse.ArgumentParser:
    p = _retrieve_parser()
    p.description = "kgqa 统一评测"
    p.add_argument("--summary", default=None, help="summary.json 输出路径")
    return p


def _gold_strings(sample, adapter, id2ent, gold_key: str) -> set[str]:
    """按 spec.gold_key 统一 gold 口径。

    - mid: gold_ids 是整数实体 id → 经 id2ent 映射成 MID，与 prediction（MID 键）/
      路径尾（id2ent[tail]=MID）同口径。
    - name: gold_ids 映射成实体名（预留给 MetaQA 等名称口径数据集）。
    """
    out: set[str] = set()
    for g in sample.gold_ids:
        if gold_key == "name":
            out.add(adapter.entity_name(str(g)))
        else:
            out.add(id2ent.get(int(g), str(g)) if isinstance(g, int) else str(g))
    return out


def main(argv=None):
    args = build_parser().parse_args(argv)
    backend, results = run_retrieval(args)
    adapter = backend.adapter
    spec = adapter.metric_spec()
    id2ent = backend.bundle.meta.id2ent
    id2rel = backend.bundle.meta.id2rel

    gold_by_index: dict[int, set[str]] = {}
    ans_records = []
    for r, sample in zip(results, backend.bundle.samples):
        gold = _gold_strings(sample, adapter, id2ent, spec.gold_key)
        gold_by_index[r.sample_index] = gold
        pred = list(r.prediction.keys())  # build_prediction 已是 MID 口径
        ans_records.append(answer_record(pred=pred, gold=sorted(gold),
                                         hop=sample.hop, format_ok=True))

    summary = {
        "answer": answer_summary(ans_records, spec),
        "path": path_summary(results, gold_by_index, spec, id2rel=id2rel),
        "n": len(results),
    }
    if args.summary:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary)), exist_ok=True)
        with open(args.summary, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(json.dumps(summary["answer"]["overall"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
