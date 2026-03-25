"""
KG-CoT 数据构造脚本（第四章核心流水线）

功能：
  1. Golden Path 标注：判断路径尾实体是否在 golden 答案中
  2. 多版本输出格式生成（V1/V2/V3/V4）
  3. 数据增强：路径顺序打乱、干扰路径比例控制、MetaQA 子集采样

输入：predict.py 输出的 JSONL（含 mmr_reason_paths 和 golden 字段）
输出：Unsloth SFT 格式的 JSONL（messages 字段，chat 格式）

用法示例：
  # WebQSP，V2 主方法，打乱路径顺序
  python llm_infer/build_kgcot_dataset.py \\
      --input  data/output/WebQSP/predict_train.jsonl \\
      --output data/output/WebQSP/kgcot_v2_train.jsonl \\
      --format v2 --shuffle

  # MetaQA 采样 20K 条
  python llm_infer/build_kgcot_dataset.py \\
      --input  data/output/MetaQA/predict_train.jsonl \\
      --output data/output/MetaQA/kgcot_v2_20k.jsonl \\
      --format v2 --shuffle --sample 20000

  # 一次生成全部四种格式（每种单独文件，用于消融实验）
  python llm_infer/build_kgcot_dataset.py \\
      --input  data/output/WebQSP/predict_train.jsonl \\
      --output data/output/WebQSP/kgcot.jsonl \\
      --format all --shuffle
"""

import argparse
import json
import logging
import os
import random
import sys
from collections import Counter
from typing import Optional

# kg_format 与本脚本同目录（llm_infer/），直接导入
from kg_format import (
    FORMAT_PROMPTS,
    build_user_content,
    format_path_str,  # noqa: F401（部分调用方可能直接用）
)


# ─── 日志 ─────────────────────────────────────────────────────────────────────

def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("build_kgcot")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ─── Golden Path 标注 ──────────────────────────────────────────────────────────

def label_paths(mmr_reason_paths: list, golden: list) -> list:
    """
    对每条路径打 Golden/Distractor 标签。
    Golden：尾实体（最后一条边的 object）在 golden 答案集中。
    返回：[(path_edges, log_score, is_golden), ...]
    """
    golden_set = {g.lower().strip() for g in golden}
    result = []
    for p in mmr_reason_paths:
        edges     = p.get("path", [])
        log_score = p.get("log_score", 0.0)
        tail      = edges[-1][2].lower().strip() if edges else None
        is_golden = tail in golden_set if tail else False
        result.append((edges, log_score, is_golden))
    return result


# ─── 数据增强 ──────────────────────────────────────────────────────────────────

def augment(labeled: list, shuffle: bool, distractor_ratio: Optional[float],
            rng: random.Random) -> list:
    """
    数据增强：
      shuffle:          完全随机打乱路径顺序，防止 positional bias
      distractor_ratio: 干扰路径占比上限（0~1），None 表示不调整

    路径顺序完全随机，不区分 golden/distractor 的位置，
    让模型学会通过路径内容判断支持路径，而非依赖位置先验。
    截断时保留 golden paths 的逻辑由 train_sft.py 在 tokenization 阶段处理。
    """
    golden_paths     = [p for p in labeled if p[2]]
    distractor_paths = [p for p in labeled if not p[2]]

    # 干扰路径比例控制
    if distractor_ratio is not None and 0 < distractor_ratio < 1:
        n_g = len(golden_paths)
        if n_g > 0 and distractor_paths:
            # 目标干扰数 = n_g * ratio / (1 - ratio)
            n_d_target = max(1, round(n_g * distractor_ratio / (1 - distractor_ratio)))
            if len(distractor_paths) > n_d_target:
                distractor_paths = rng.sample(distractor_paths, n_d_target)

    result = golden_paths + distractor_paths
    if shuffle:
        rng.shuffle(result)  # 完全随机打乱，golden 和 distractor 混合
    return result


# ─── 输出格式 ──────────────────────────────────────────────────────────────────

def output_v1(answers: list) -> str:
    """V1 SFT-Answer：仅输出答案。"""
    return "Answer: " + " | ".join(answers)


def output_v2(golden_indices: list, answers: list) -> str:
    """V2 SFT-Cite（主方法）：纯文本路径引用 + 答案。"""
    cited = ", ".join(str(i) for i in sorted(golden_indices))
    return f"Supporting Paths: {cited}\nAnswer: {' | '.join(answers)}"


def output_v3(golden_indices: list, answers: list) -> str:
    """V3 SFT-JSON：JSON 格式，消融对比项。"""
    return json.dumps(
        {"reasoning": [f"Path {i}" for i in sorted(golden_indices)],
         "answer":    answers},
        ensure_ascii=False,
    )


def output_v4(paths_with_meta: list, golden_indices: list, answers: list) -> str:
    """V4 SFT-CoT：自然语言推理链 + 答案，消融对比项。"""
    golden_set = set(golden_indices)
    reasoning_lines = []
    for edges, log_score, display_idx in paths_with_meta:
        if display_idx in golden_set and edges:
            desc = " -> ".join(f"({e[0]}) -[{e[1]}]-> ({e[2]})" for e in edges)
            tail = edges[-1][2]
            reasoning_lines.append(
                f"Path {display_idx} supports '{tail}': {desc}"
            )
    reasoning = "\n".join(reasoning_lines) if reasoning_lines else "No supporting path found."
    cited = ", ".join(str(i) for i in sorted(golden_indices))
    return (
        f"[Reasoning]\n{reasoning}\n\n"
        f"[Answer]\n"
        f"Supporting Paths: {cited}\n"
        f"Answer: {' | '.join(answers)}"
    )


# ─── 单样本构造 ────────────────────────────────────────────────────────────────

def make_sample(record: dict, fmt: str, shuffle: bool,
                distractor_ratio: Optional[float], show_score: bool,
                rng: random.Random) -> Optional[dict]:
    """
    从一条 predict JSONL 记录构造训练样本。
    返回 None 表示 Hit@K 未命中，丢弃。

    show_score: 路径字符串中是否包含 [score=S]（False 用于消融实验）
    """
    question = record.get("question", "")
    mmr_paths = record.get("mmr_reason_paths", [])
    golden    = record.get("golden", [])

    if not question or not golden:
        return None

    labeled = label_paths(mmr_paths, golden)

    # Hit@K 检查：必须至少有一条 Golden Path
    if not any(is_g for _, _, is_g in labeled):
        return None

    labeled = augment(labeled, shuffle, distractor_ratio, rng)

    # 重新分配 display index（1-based）
    paths_with_meta = [
        (edges, score, i + 1)
        for i, (edges, score, _) in enumerate(labeled)
    ]
    is_golden_flags = [is_g for _, _, is_g in labeled]
    golden_indices  = [i + 1 for i, is_g in enumerate(is_golden_flags) if is_g]

    # 答案使用路径中 tail entity 的原文（保证与路径内容忠实一致）
    # 若与 golden 列表存在大小写差异，以路径为准；保序去重
    path_answers = list(dict.fromkeys(
        edges[-1][2] for edges, _, is_g in labeled if is_g and edges
    ))
    # 极端情况兜底（理论上不应发生）
    answer_entities = path_answers if path_answers else golden

    user_content = build_user_content(paths_with_meta, question, show_score=show_score)
    system_prompt = FORMAT_PROMPTS.get(fmt, FORMAT_PROMPTS["v2"])

    if fmt == "v1":
        asst = output_v1(answer_entities)
    elif fmt == "v2":
        asst = output_v2(golden_indices, answer_entities)
    elif fmt == "v3":
        asst = output_v3(golden_indices, answer_entities)
    elif fmt == "v4":
        asst = output_v4(paths_with_meta, golden_indices, answer_entities)
    else:
        raise ValueError(f"未知格式: {fmt}")

    return {
        "messages": [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": asst},
        ],
        "_meta": {
            "question":            question,
            "golden":              golden,
            "path_answers":        answer_entities,
            "golden_path_indices": golden_indices,
            "n_golden":            len(golden_indices),
            "n_distractor":        len(labeled) - len(golden_indices),
            "format":              fmt,
            "show_score":          show_score,
            "hop":                 record.get("hop"),
        },
    }


# ─── 核心流程 ──────────────────────────────────────────────────────────────────

def build(input_path: str, output_path: str, fmt: str, shuffle: bool,
          distractor_ratio: Optional[float], sample_n: int, show_score: bool,
          rng: random.Random, log: logging.Logger) -> dict:
    with open(input_path, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]
    log.info("读入 %d 条记录  格式=%s  show_score=%s", len(records), fmt, show_score)

    if sample_n > 0 and len(records) > sample_n:
        records = rng.sample(records, sample_n)
        log.info("采样后 %d 条", len(records))

    samples, skipped = [], 0
    for rec in records:
        s = make_sample(rec, fmt, shuffle, distractor_ratio, show_score, rng)
        if s is None:
            skipped += 1
        else:
            samples.append(s)

    log.info("有效样本 %d  丢弃(Hit@K=0) %d", len(samples), skipped)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    log.info("输出: %s", output_path)

    n = len(samples)
    if n == 0:
        return {"format": fmt, "total": 0, "skipped": skipped,
                "avg_golden": 0, "avg_distractor": 0}

    # ── 数据质量统计 ──────────────────────────────────────────────────────────
    avg_golden     = round(sum(s["_meta"]["n_golden"]     for s in samples) / n, 2)
    avg_distractor = round(sum(s["_meta"]["n_distractor"] for s in samples) / n, 2)

    # 序列长度估算（字符数 / 4 ≈ token 数）
    seq_lens = [
        (len(s["messages"][1]["content"]) + len(s["messages"][2]["content"])) // 4
        for s in samples
    ]
    seq_lens_sorted = sorted(seq_lens)
    p90_idx = int(len(seq_lens_sorted) * 0.9)
    log.info("序列长度估算(token): min=%d  avg=%d  p90=%d  max=%d",
             seq_lens_sorted[0],
             sum(seq_lens) // n,
             seq_lens_sorted[p90_idx],
             seq_lens_sorted[-1])

    # Per-hop 分布
    hop_dist = Counter(str(s["_meta"].get("hop", "?")) for s in samples)
    if len(hop_dist) > 1:
        hop_str = "  ".join(f"hop={k}:{v}" for k, v in sorted(hop_dist.items()))
        log.info("跳数分布: %s", hop_str)

    return {
        "format":         fmt,
        "total":          n,
        "skipped":        skipped,
        "avg_golden":     avg_golden,
        "avg_distractor": avg_distractor,
        "seq_len_avg":    sum(seq_lens) // n,
        "seq_len_p90":    seq_lens_sorted[p90_idx],
        "seq_len_max":    seq_lens_sorted[-1],
        "hop_dist":       dict(hop_dist),
    }


# ─── 主入口 ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="KG-CoT 数据构造")
    p.add_argument("--input",  required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--format", default="v2",
                   choices=["v1", "v2", "v3", "v4", "all"],
                   help="v1=仅答案 v2=路径引用(主方法) v3=JSON v4=CoT all=全部四种")
    p.add_argument("--no_shuffle", action="store_true",
                   help="关闭路径顺序随机打乱（默认开启，用于防止 positional bias）")
    p.add_argument("--no_score", action="store_true",
                   help="路径字符串中不包含 score（消融实验用）")
    p.add_argument("--distractor_ratio", type=float, default=None,
                   help="干扰路径占比上限 0~1，None=不调整")
    p.add_argument("--sample", type=int, default=0,
                   help="采样 N 条（0=全量），MetaQA 329K 时使用")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng  = random.Random(args.seed)

    shuffle    = not args.no_shuffle
    show_score = not args.no_score

    log_path = os.path.splitext(args.output)[0] + "_build.log"
    log = setup_logger(log_path)
    log.info("命令: %s", " ".join(sys.argv))
    log.info("shuffle=%s  show_score=%s", shuffle, show_score)

    if args.format == "all":
        base, ext = os.path.splitext(args.output)
        ext = ext or ".jsonl"
        stats_list = []
        for fmt in ["v1", "v2", "v3", "v4"]:
            stat = build(args.input, f"{base}_{fmt}{ext}", fmt,
                         shuffle, args.distractor_ratio, args.sample,
                         show_score, rng, log)
            stats_list.append(stat)
        log.info("=" * 50)
        for st in stats_list:
            log.info("[%s] total=%d skip=%d avg_golden=%.2f avg_distractor=%.2f"
                     " seq_len_avg=%d seq_len_p90=%d",
                     st["format"], st["total"], st["skipped"],
                     st["avg_golden"], st["avg_distractor"],
                     st.get("seq_len_avg", 0), st.get("seq_len_p90", 0))
    else:
        st = build(args.input, args.output, args.format,
                   shuffle, args.distractor_ratio, args.sample,
                   show_score, rng, log)
        log.info("total=%d skip=%d avg_golden=%.2f avg_distractor=%.2f"
                 " seq_len_avg=%d seq_len_p90=%d",
                 st["total"], st["skipped"],
                 st["avg_golden"], st["avg_distractor"],
                 st.get("seq_len_avg", 0), st.get("seq_len_p90", 0))

    log.info("日志: %s", log_path)


if __name__ == "__main__":
    main()
