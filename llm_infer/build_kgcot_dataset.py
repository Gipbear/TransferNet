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
from typing import Optional


# ─── Prompt ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a KGQA assistant. "
    "Given reasoning paths from a knowledge graph and a question, "
    "identify which paths support the answer, then extract the answer "
    "from the tail entities of those supporting paths.\n"
    "Rules:\n"
    "- Only output entity IDs that appear in the provided paths.\n"
    "- Do not generate or fabricate new entity IDs.\n"
    "Output format:\n"
    "Supporting Paths: <path numbers>\n"
    "Answer: <entity_id> | <entity_id>"
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


# ─── 路径格式化 ────────────────────────────────────────────────────────────────

def format_path_str(path_edges: list, log_score: float, idx: int) -> str:
    """将路径序列化为 'Path N [score=S]: (e0) -[r]-> (e1) ...' 字符串。"""
    chain = " ".join(f"({e[0]}) -[{e[1]}]-> ({e[2]})" for e in path_edges)
    return f"Path {idx} [score={log_score:.4f}]: {chain}"


def build_user_content(paths_with_meta: list, question: str) -> str:
    """
    构建 User 消息。问题前置，让模型带着目标扫描路径。
    paths_with_meta: [(path_edges, log_score, display_idx), ...]
    """
    lines = [f"Question: {question}", "", "Reasoning Paths:"]
    for path_edges, log_score, display_idx in paths_with_meta:
        lines.append(format_path_str(path_edges, log_score, display_idx))
    return "\n".join(lines)


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
      shuffle:          随机打乱路径顺序，防止 positional bias
      distractor_ratio: 干扰路径占比上限（0~1），None 表示不调整
    """
    result = list(labeled)

    # 干扰路径比例控制
    if distractor_ratio is not None and 0 < distractor_ratio < 1:
        golden_paths     = [p for p in result if p[2]]
        distractor_paths = [p for p in result if not p[2]]
        n_g = len(golden_paths)
        if n_g > 0 and distractor_paths:
            # 目标干扰数 = n_g * ratio / (1 - ratio)
            n_d_target = max(1, round(n_g * distractor_ratio / (1 - distractor_ratio)))
            if len(distractor_paths) > n_d_target:
                distractor_paths = rng.sample(distractor_paths, n_d_target)
            result = golden_paths + distractor_paths

    if shuffle:
        rng.shuffle(result)

    return result


# ─── 输出格式 ──────────────────────────────────────────────────────────────────

def output_v1(golden_answers: list) -> str:
    """V1 SFT-Answer：仅输出答案（与 V0 零样本格式相同，但经微调）。"""
    return "Answer: " + " | ".join(golden_answers)


def output_v2(golden_indices: list, golden_answers: list) -> str:
    """V2 SFT-Cite（主方法）：纯文本路径引用 + 答案。"""
    cited = ", ".join(str(i) for i in sorted(golden_indices))
    return f"Supporting Paths: {cited}\nAnswer: {' | '.join(golden_answers)}"


def output_v3(golden_indices: list, golden_answers: list) -> str:
    """V3 SFT-JSON：JSON 格式，消融对比项。"""
    return json.dumps(
        {"reasoning": [f"Path {i}" for i in sorted(golden_indices)],
         "answer":    golden_answers},
        ensure_ascii=False,
    )


def output_v4(paths_with_meta: list, golden_indices: list, golden_answers: list) -> str:
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
        f"Answer: {' | '.join(golden_answers)}"
    )


# ─── 单样本构造 ────────────────────────────────────────────────────────────────

def make_sample(record: dict, fmt: str, shuffle: bool,
                distractor_ratio: Optional[float], rng: random.Random) -> Optional[dict]:
    """
    从一条 predict JSONL 记录构造训练样本。
    返回 None 表示 Hit@K 未命中，丢弃。
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

    user_content = build_user_content(paths_with_meta, question)

    if fmt == "v1":
        asst = output_v1(golden)
    elif fmt == "v2":
        asst = output_v2(golden_indices, golden)
    elif fmt == "v3":
        asst = output_v3(golden_indices, golden)
    elif fmt == "v4":
        asst = output_v4(paths_with_meta, golden_indices, golden)
    else:
        raise ValueError(f"未知格式: {fmt}")

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": asst},
        ],
        "_meta": {
            "question":              question,
            "golden":                golden,
            "golden_path_indices":   golden_indices,
            "n_golden":              len(golden_indices),
            "n_distractor":          len(labeled) - len(golden_indices),
            "format":                fmt,
            "hop":                   record.get("hop"),
        },
    }


# ─── 核心流程 ──────────────────────────────────────────────────────────────────

def build(input_path: str, output_path: str, fmt: str, shuffle: bool,
          distractor_ratio: Optional[float], sample_n: int,
          rng: random.Random, log: logging.Logger) -> dict:
    with open(input_path, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]
    log.info("读入 %d 条记录  格式=%s", len(records), fmt)

    if sample_n > 0 and len(records) > sample_n:
        records = rng.sample(records, sample_n)
        log.info("采样后 %d 条", len(records))

    samples, skipped = [], 0
    for rec in records:
        s = make_sample(rec, fmt, shuffle, distractor_ratio, rng)
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
    return {
        "format":      fmt,
        "total":       n,
        "skipped":     skipped,
        "avg_golden":  round(sum(s["_meta"]["n_golden"]    for s in samples) / n, 2) if n else 0,
        "avg_distractor": round(sum(s["_meta"]["n_distractor"] for s in samples) / n, 2) if n else 0,
    }


# ─── 主入口 ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="KG-CoT 数据构造")
    p.add_argument("--input",  required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--format", default="v2",
                   choices=["v1", "v2", "v3", "v4", "all"],
                   help="v1=仅答案 v2=路径引用(主方法) v3=JSON v4=CoT all=全部四种")
    p.add_argument("--shuffle", action="store_true",
                   help="随机打乱路径顺序（推荐开启，防止 positional bias）")
    p.add_argument("--distractor_ratio", type=float, default=None,
                   help="干扰路径占比上限 0~1，None=不调整")
    p.add_argument("--sample", type=int, default=0,
                   help="采样 N 条（0=全量），MetaQA 329K 时使用")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng  = random.Random(args.seed)

    log_path = os.path.splitext(args.output)[0] + "_build.log"
    log = setup_logger(log_path)
    log.info("命令: %s", " ".join(sys.argv))

    if args.format == "all":
        base, ext = os.path.splitext(args.output)
        ext = ext or ".jsonl"
        stats_list = []
        for fmt in ["v1", "v2", "v3", "v4"]:
            stat = build(args.input, f"{base}_{fmt}{ext}", fmt,
                         args.shuffle, args.distractor_ratio, args.sample, rng, log)
            stats_list.append(stat)
        log.info("=" * 50)
        for st in stats_list:
            log.info("[%s] total=%d skip=%d avg_golden=%.2f avg_distractor=%.2f",
                     st["format"], st["total"], st["skipped"],
                     st["avg_golden"], st["avg_distractor"])
    else:
        st = build(args.input, args.output, args.format,
                   args.shuffle, args.distractor_ratio, args.sample, rng, log)
        log.info("total=%d skip=%d avg_golden=%.2f avg_distractor=%.2f",
                 st["total"], st["skipped"], st["avg_golden"], st["avg_distractor"])

    log.info("日志: %s", log_path)


if __name__ == "__main__":
    main()
