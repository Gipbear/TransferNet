"""按 hop 分层采样生成子集 qa 文件(与源同格式)。

MetaQA train 329K 全量 dump 不可行(约 6GB);先分层采子集,
`kgqa.retrieve.cli.dump_scores --qa_file <子集>` 原样使用,dump/producer 零改动。

支持两种源格式(自动识别,输出与源同格式):
- JSON 数组(带 hop 字段的原始 qa 文件)
- MetaQA_KB 预处理 .pt(questions/topic_entities/answers/hops 四段 pickle,
  `MetaQA_KB.data.DataLoader` 的实际输入;四数组按同一索引切片保持对齐)

用法:
  python -m kgqa.pfit.subset_qa \\
      --input data/input/MetaQA_KB/train.pt \\
      --output data/output/kgqa/metaqa/subsets/train_20k.pt \\
      --n 20000 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
from collections import Counter

from kgqa.pfit.build import stratified_sample_by_hop

log = logging.getLogger("pfit.subset_qa")


def _is_json_array(path: str) -> bool:
    with open(path, "rb") as f:
        head = f.read(64).lstrip()
    return head.startswith(b"[")


def _subset_json(input_path: str, output_path: str, *, n: int, seed: int) -> list:
    with open(input_path, encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError(f"{input_path} 不是 JSON 数组 qa 文件")

    if 0 < n < len(items):
        subset = stratified_sample_by_hop(items, n, random.Random(seed))
    else:
        subset = items

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False)

    dist = Counter(str(x.get("hop", "?")) for x in subset)
    log.info("JSON 子集 %d/%d 条  跳数分布 %s  → %s",
             len(subset), len(items), dict(sorted(dist.items())), output_path)
    return subset


def _subset_question_pt(input_path: str, output_path: str, *, n: int, seed: int):
    """MetaQA_KB 预处理 .pt:按 hops 分层采索引,四数组同索引切片。

    pickle 仅用于读写本地 `MetaQA_KB.preprocess` 产物(与 MetaQA_KB/data.py
    同一信任模型),不接受外部来源文件。
    """
    inputs = []
    with open(input_path, "rb") as f:
        for _ in range(4):
            inputs.append(pickle.load(f))
    questions, topic_entities, answers, hops = inputs
    total = len(questions)
    if not (len(topic_entities) == len(answers) == len(hops) == total):
        raise ValueError(f"{input_path} 四段数组长度不一致")

    if 0 < n < total:
        index_recs = [{"idx": i, "hop": int(h)} for i, h in enumerate(hops)]
        sampled = stratified_sample_by_hop(index_recs, n, random.Random(seed))
        idx = sorted(r["idx"] for r in sampled)  # 保持源相对顺序
        subset = [questions[idx], topic_entities[idx], answers[idx],
                  [int(hops[i]) for i in idx] if isinstance(hops, list) else hops[idx]]
    else:
        idx = list(range(total))
        subset = inputs

    with open(output_path, "wb") as f:
        for arr in subset:
            pickle.dump(arr, f)

    dist = Counter(str(int(subset[3][i])) for i in range(len(idx)))
    log.info(".pt 子集 %d/%d 条  跳数分布 %s  → %s",
             len(idx), total, dict(sorted(dist.items())), output_path)
    return subset


def make_subset(input_path: str, output_path: str, *, n: int, seed: int = 42):
    """按 hop 分层采 n 条(配额按原分布),输出与源同格式。"""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if _is_json_array(input_path):
        return _subset_json(input_path, output_path, n=n, seed=seed)
    return _subset_question_pt(input_path, output_path, n=n, seed=seed)


def build_parser():
    p = argparse.ArgumentParser(description="按 hop 分层采样生成子集 qa 文件")
    p.add_argument("--input", required=True,
                   help="源 qa 文件(JSON 数组或 MetaQA_KB 预处理 .pt)")
    p.add_argument("--output", required=True)
    p.add_argument("--n", type=int, required=True, help="采样条数(≥总量时全保留)")
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv=None):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    a = build_parser().parse_args(argv)
    make_subset(a.input, a.output, n=a.n, seed=a.seed)


if __name__ == "__main__":
    main()
