"""构造 strict reject-list checker adapter 的 SFT 训练集。

为每个 WebQSP TRAIN 问题:从 train score cache 检索 beam 路径,按批切分,
构造流水线实际使用的 strict-check 提示词,并按 golden 答案标注各候选
答案索引是否应被 reject。输出 messages 格式 JSONL,供 kgqa.pfit.train
训练 checker adapter(经 llm_server 挂载后由 checked-batch 流水的
reject 检查消费)。

检索经 kgqa.retrieve.api.service.PathRetrieveService——即 path_retrieve_server
进程内同款实现,与评测时 checker 实际看到的路径输入同源同参数。

迁自 llm_infer/build_checker_dataset.py(原依赖 oh_my_agent,其检索/实体映射
底层本就来自 kgqa,迁移仅替换 import 与构造方式)。

前置条件(参见 docs/experiments/experiments_checked_batch_optimization_202606.md §7):
  1. 先把 train QA 过滤为 loader 实际会加载的行:
       python scripts/build_webqsp_fixed_qa.py \
           --source data/input/WebQSP/QA_data/WebQuestionsSP/qa_train_webqsp.txt \
           --output data/input/WebQSP/QA_data/WebQuestionsSP/qa_train_webqsp_fixed.txt
  2. 用同一(过滤后)qa 文件 dump train score cache。cache 的问题文本必须与
     本脚本逐字匹配,否则 golden 标签整体错位;dump 时须避开默认被 shuffle
     的 train loader(否则问题与得分张量静默错位)。历史产物由
     WebQSP.dump_scores 以 --mode val + 绝对 --qa_file 生成;现役统一入口为
     kgqa.retrieve.cli.dump_scores,具体参数以该 CLI 的 --help 为准。

用法:
  python -m kgqa.agent.cli.build_checker_dataset \
      --cache <TRAIN_CACHE_PT> --qa_file <FIXED_TRAIN_QA> --out <OUT_JSONL>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from kgqa.agent.common import apply_entity_map, load_entity_map, load_webqsp_qa_samples
from kgqa.agent.specs import get_agent_spec
from kgqa.agent.tools.cited_path_check import (
    STRICT_REJECTED_ANSWER_CHECK_SYSTEM,
    _candidate_answers_from_cited_paths,
    build_rejected_answer_prompt,
)
from kgqa.retrieve.api.service import PathRetrieveService
from kgqa.retrieve.datasets.registry import get_adapter

_DATASET = "webqsp"


def _norm(value: str) -> str:
    return str(value).lower().strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cache",
        default="data/output/WebQSP/path_retrieve_server/score_cache/webqsp_train_2996_fixed.pt",
        help="Train score cache from dump_scores (see module docstring)",
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
    parser.add_argument(
        "--entity_map",
        default=None,
        help="实体映射文件路径 (MID→Name, tab-separated);缺省用 webqsp agent spec 的默认映射",
    )
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
    adapter = get_adapter(_DATASET, input_dir=args.input_dir)
    retriever = PathRetrieveService(adapter, cache_path=args.cache)
    if args.entity_map:
        entity_map = load_entity_map(args.entity_map)
    else:
        entity_map = get_agent_spec(_DATASET).load_entity_map()

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
