"""Build an SFT dataset for a strict reject-list checker adapter.

For each WebQSP TRAIN question: retrieve beam paths from a train score cache
(same MMR settings as the eval pipeline), split into batches, build the exact
strict-check prompt the pipeline uses, and label rejected candidate indices by
gold answers. Output is messages-format JSONL for llm_infer.train_sft.

Prerequisites (see docs/experiments_checked_batch_optimization_202606.md §7):
  1. Filter the train QA file to loader-valid rows:
       python scripts/build_webqsp_fixed_qa.py \
           --source data/input/WebQSP/QA_data/WebQuestionsSP/qa_train_webqsp.txt \
           --output data/input/WebQSP/QA_data/WebQuestionsSP/qa_train_webqsp_fixed.txt
  2. Dump the train score cache with the SAME (filtered) file. Use an ABSOLUTE
     --qa_file path and --mode val (NOT train: that selects the shuffled default
     train loader and silently misaligns questions with score tensors):
       python -m WebQSP.dump_scores --input_dir data/input/WebQSP \
           --ckpt <CKPT> --mode val --qa_file <ABS_PATH_TO_FIXED_TRAIN_QA> \
           --output <TRAIN_CACHE_PT>

Usage:
  python -m llm_infer.build_checker_dataset \
      --cache <TRAIN_CACHE_PT> --qa_file <FIXED_TRAIN_QA> --out <OUT_JSONL>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oh_my_agent.common import apply_entity_map, load_entity_map
from oh_my_agent.common.qa_data import load_webqsp_qa_samples
from oh_my_agent.path_retrieve_server.service import CachedPathRetriever
from oh_my_agent.tools.cited_path_check import (
    STRICT_REJECTED_ANSWER_CHECK_SYSTEM,
    _candidate_answers_from_cited_paths,
    build_rejected_answer_prompt,
)
from oh_my_agent.tools.path_retrieve import DEFAULT_ENTITY_MAP_PATH


def _norm(value: str) -> str:
    return str(value).lower().strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cache",
        default="data/output/WebQSP/path_retrieve_server/score_cache/webqsp_train_2996_fixed.pt",
        help="Train score cache from WebQSP.dump_scores (see module docstring)",
    )
    parser.add_argument("--input_dir", default="data/input/WebQSP")
    parser.add_argument(
        "--qa_file",
        default="data/input/WebQSP/QA_data/WebQuestionsSP/qa_train_webqsp_fixed.txt",
        help="Loader-aligned QA file (build_webqsp_fixed_qa.py output)",
    )
    parser.add_argument(
        "--out",
        default="data/output/WebQSP/llm_dataset/checker_strict_v1/checker_train.jsonl",
    )
    parser.add_argument("--entity_map", default=DEFAULT_ENTITY_MAP_PATH)
    parser.add_argument("--beam_size", type=int, default=50)
    parser.add_argument("--lambda_val", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument(
        "--max_batches",
        type=int,
        default=2,
        help="Path batches per question turned into training examples",
    )
    parser.add_argument("--limit", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    samples = load_webqsp_qa_samples(args.qa_file, limit=args.limit)
    retriever = CachedPathRetriever(cache_path=args.cache, input_dir=args.input_dir)
    entity_map = load_entity_map(args.entity_map)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_records = n_skipped = n_all_reject = n_none = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for i, sample in enumerate(samples):
            try:
                result = retriever.retrieve(
                    question=sample.question,
                    beam_size=args.beam_size,
                    lambda_val=args.lambda_val,
                )
            except Exception:
                n_skipped += 1
                continue
            # 问题文本必须与缓存逐字匹配,否则标签会整体错位
            if _norm(result.question) != _norm(sample.question):
                n_skipped += 1
                continue

            gold = {_norm(mid) for mid in sample.gold_mids}
            raw_paths = result.mmr_reason_paths
            named_paths = [
                {
                    "path": apply_entity_map(p.get("path", []), entity_map),
                    "log_score": p.get("log_score", 0.0),
                }
                for p in raw_paths
            ]
            for batch_index in range(args.max_batches):
                start = batch_index * args.batch_size
                batch_named = named_paths[start : start + args.batch_size]
                batch_raw = raw_paths[start : start + args.batch_size]
                if not batch_named:
                    break
                cited = list(range(1, len(batch_named) + 1))
                candidates, cited_paths = _candidate_answers_from_cited_paths(
                    batch_named, batch_raw, cited
                )
                if not candidates:
                    continue
                prompt = build_rejected_answer_prompt(
                    sample.question,
                    cited_paths,
                    candidates,
                    strict=True,
                )
                rejected = sorted(
                    c["index"]
                    for c in candidates
                    if _norm(str(c.get("mid", ""))) not in gold
                )
                label = ",".join(str(x) for x in rejected) if rejected else "NONE"
                if not rejected:
                    n_none += 1
                elif len(rejected) == len(candidates):
                    n_all_reject += 1
                golden_paths = sorted(
                    {
                        path_index
                        for c in candidates
                        if _norm(str(c.get("mid", ""))) in gold
                        for path_index in c.get("path_indices", [])
                    }
                )
                record = {
                    "messages": [
                        {"role": "system", "content": STRICT_REJECTED_ANSWER_CHECK_SYSTEM},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": label},
                    ],
                    "_meta": {
                        "question": sample.question,
                        "golden": sample.gold_mids,
                        "golden_path_indices": golden_paths,
                        "batch_index": batch_index,
                        "n_candidates": len(candidates),
                        "n_rejected": len(rejected),
                    },
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_records += 1
            if (i + 1) % 500 == 0:
                print(f"[{i + 1}/{len(samples)}] records={n_records}", flush=True)

    print(
        f"DONE records={n_records} skipped_samples={n_skipped} "
        f"all_reject={n_all_reject} none={n_none}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
