"""统一评测 CLI：检索 + 答案级/路径级指标。"""
from __future__ import annotations

import argparse
import json
import os

from kgqa.retrieve.cli.retrieve import build_parser as _retrieve_parser, run_retrieval
from kgqa.retrieve.eval.answer_eval import answer_record, answer_summary
from kgqa.retrieve.eval.path_eval import path_summary
from kgqa.runtime import emit_event, update_progress


def build_parser() -> argparse.ArgumentParser:
    p = _retrieve_parser()
    p.description = "kgqa 统一评测"
    p.add_argument("--summary", default=None, help="summary.json 输出路径")
    return p


def _gold_strings(sample, adapter, id2ent, gold_key: str) -> set[str]:
    """按 spec.gold_key 统一 gold 口径。

    - mid: gold_ids 是整数实体 id → 经 id2ent 映射成 MID，与 prediction（MID 键）/
      路径尾（id2ent[tail]=MID）同口径。
    - name: 整数 gold_ids 同样先经 id2ent 还原（MetaQA id2ent 即实体名），再过
      adapter.entity_name，与 prediction（名称键）同口径。
    """
    out: set[str] = set()
    for g in sample.gold_ids:
        base = id2ent.get(int(g), str(g)) if isinstance(g, int) else str(g)
        out.add(adapter.entity_name(base) if gold_key == "name" else base)
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
    run_dir = getattr(args, "run_dir", "") or (os.path.dirname(os.path.abspath(args.summary)) if args.summary else "")
    update_progress(run_dir, completed=len(results), total=len(results), status="completed", phase="检索评测")
    emit_event(run_dir, "phase_end", phase="检索评测", samples=len(results))
    print(json.dumps(summary["answer"]["overall"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
