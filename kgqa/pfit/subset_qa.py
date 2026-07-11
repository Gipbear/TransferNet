"""按 hop 分层采样生成子集 qa 文件(JSON 数组,与源同格式)。

MetaQA train 329K 全量 dump 不可行(约 6GB);先分层采子集,
`kgqa.cli.dump_scores --qa_file <子集>` 原样使用,dump/producer 零改动。

用法:
  python -m kgqa.pfit.subset_qa \\
      --input data/input/MetaQA_KB/train.json \\
      --output data/output/kgqa/metaqa/subsets/train_20k.json \\
      --n 20000 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
from collections import Counter

from kgqa.pfit.build import stratified_sample_by_hop

log = logging.getLogger("pfit.subset_qa")


def make_subset(input_path: str, output_path: str, *, n: int, seed: int = 42) -> list:
    """读 JSON 数组 qa 文件,按 hop 分层采 n 条(配额按原分布),原样写出。"""
    with open(input_path, encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError(f"{input_path} 不是 JSON 数组 qa 文件")

    if 0 < n < len(items):
        subset = stratified_sample_by_hop(items, n, random.Random(seed))
    else:
        subset = items

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False)

    dist = Counter(str(x.get("hop", "?")) for x in subset)
    log.info("子集 %d/%d 条  跳数分布 %s  → %s",
             len(subset), len(items), dict(sorted(dist.items())), output_path)
    return subset


def build_parser():
    p = argparse.ArgumentParser(description="按 hop 分层采样生成子集 qa 文件")
    p.add_argument("--input", required=True, help="源 qa 文件(JSON 数组,带 hop 字段)")
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
