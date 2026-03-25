"""
消融实验结果汇总脚本

扫描 data/output/WebQSP/ablation/ 下所有 *_eval.log 文件，
解析关键指标，输出分组对比表格（CSV + 终端格式化）。

用法：
  python scripts/collect_ablation_results.py
  python scripts/collect_ablation_results.py --ablation_dir data/output/WebQSP/ablation
  python scripts/collect_ablation_results.py \\
      --baseline_log data/output/WebQSP/grid_search/paths/beam20_lam0.2_v2_ft_eval.log
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path


# ─── 指标解析 ──────────────────────────────────────────────────────────────────

# 日志格式参照 eval_faithfulness.py:log_metrics()
_METRIC_PATTERNS = {
    "hit1":                re.compile(r"Hit@1\s*:\s*([\d.]+)"),
    "hit_any":             re.compile(r"Hit@Any\s*:\s*([\d.]+)"),
    "macro_p":             re.compile(r"Macro\s+P/R/F1\s*:\s*([\d.]+)\s*/\s*([\d.]+)\s*/\s*([\d.]+)"),
    "micro_p":             re.compile(r"Micro\s+P/R/F1\s*:\s*([\d.]+)\s*/\s*([\d.]+)\s*/\s*([\d.]+)"),
    "exact_match":         re.compile(r"Exact Match\s*:\s*([\d.]+)"),
    "citation_accuracy":   re.compile(r"Citation Accuracy\s*:\s*([\d.]+)"),
    "citation_recall":     re.compile(r"Citation Recall\s*:\s*([\d.]+)"),
    "hallucination_rate":  re.compile(r"Hallucination Rate\s*:\s*([\d.]+)"),
    "format_compliance":   re.compile(r"Format Compliance\s*:\s*([\d.]+)"),
    "n":                   re.compile(r"\[ ALL.*?\]\s+\(n=(\d+)\)"),
}


def parse_eval_log(log_path: str) -> dict | None:
    """
    解析 eval_faithfulness.py 生成的日志文件，提取 ALL 段的指标。
    返回指标 dict，失败返回 None。
    """
    try:
        text = Path(log_path).read_text(encoding="utf-8")
    except OSError:
        return None

    # 只取 ALL 段（第一次出现的 [ ALL ... ] 块，避免子集数据污染）
    all_start = text.find("[ ALL")
    if all_start == -1:
        return None

    # 截取到下一个分层段之前（"---"分隔符或文件结尾）
    next_section = text.find("---", all_start)
    segment = text[all_start:next_section] if next_section != -1 else text[all_start:]

    result = {}

    m = _METRIC_PATTERNS["n"].search(segment)
    result["n"] = int(m.group(1)) if m else None

    for key in ("hit1", "hit_any", "exact_match",
                "citation_accuracy", "citation_recall",
                "hallucination_rate", "format_compliance"):
        m = _METRIC_PATTERNS[key].search(segment)
        result[key] = float(m.group(1)) if m else None

    m = _METRIC_PATTERNS["macro_p"].search(segment)
    if m:
        result["macro_p"]  = float(m.group(1))
        result["macro_r"]  = float(m.group(2))
        result["macro_f1"] = float(m.group(3))
    else:
        result["macro_p"] = result["macro_r"] = result["macro_f1"] = None

    m = _METRIC_PATTERNS["micro_p"].search(segment)
    if m:
        result["micro_p"]  = float(m.group(1))
        result["micro_r"]  = float(m.group(2))
        result["micro_f1"] = float(m.group(3))
    else:
        result["micro_p"] = result["micro_r"] = result["micro_f1"] = None

    return result


# ─── 日志发现 ──────────────────────────────────────────────────────────────────

def discover_logs(ablation_dir: str) -> list[dict]:
    """
    扫描 ablation_dir 下所有 *_eval.log，返回 records：
      [{"config": ..., "group": ..., "log_path": ..., "test_stem": ..., "fmt": ...}, ...]
    """
    records = []
    ablation_path = Path(ablation_dir)
    if not ablation_path.exists():
        return records

    for log_file in sorted(ablation_path.rglob("*_eval.log")):
        config_name = log_file.parent.name  # e.g. groupA_v1, groupC
        group = classify_group(config_name)

        # 从文件名解析 test_stem 和 format
        # 命名规则: {test_stem}_{fmt}_ft_eval.log
        # 例: beam20_lam0.2_v2_ft_eval.log
        stem = log_file.stem  # beam20_lam0.2_v2_ft_eval
        if not stem.endswith("_ft_eval"):
            continue
        inner = stem[: -len("_ft_eval")]  # beam20_lam0.2_v2
        # 最后一个 _ 前为 test_stem，后为 fmt
        parts = inner.rsplit("_", 1)
        if len(parts) != 2:
            continue
        test_stem, fmt = parts[0], parts[1]

        records.append({
            "config":    config_name,
            "group":     group,
            "log_path":  str(log_file),
            "test_stem": test_stem,
            "fmt":       fmt,
        })

    return records


def classify_group(config_name: str) -> str:
    if config_name.startswith("groupA"):
        return "A"
    if config_name.startswith("groupB"):
        return "B"
    if config_name.startswith("groupC"):
        return "C"
    return "X"


# ─── 表格格式化 ────────────────────────────────────────────────────────────────

def fmt_val(v, precision: int = 4) -> str:
    if v is None:
        return "--"
    return f"{v:.{precision}f}"


def print_group_a(rows: list[dict]):
    """Group A: 输出格式消融对比表"""
    print("\n" + "=" * 70)
    print("  Group A: 输出格式消融 (测试集: beam20_lam0.2)")
    print("=" * 70)
    header = f"{'Format':<10} {'Hit@1':>7} {'MacroF1':>8} {'EM':>7} {'CitAcc':>7} {'CitRec':>7} {'Hallu':>7} {'n':>6}"
    print(header)
    print("-" * 70)
    for r in rows:
        m = r.get("metrics") or {}
        is_v1 = r.get("fmt") == "v1"
        cit_acc = "--" if is_v1 else fmt_val(m.get("citation_accuracy"))
        cit_rec = "--" if is_v1 else fmt_val(m.get("citation_recall"))
        tag = " *" if r.get("config") == "groupA_v2" else "  "
        print(
            f"{r.get('fmt','?'):<10}"
            f" {fmt_val(m.get('hit1')):>7}"
            f" {fmt_val(m.get('macro_f1')):>8}"
            f" {fmt_val(m.get('exact_match')):>7}"
            f" {cit_acc:>7}"
            f" {cit_rec:>7}"
            f" {fmt_val(m.get('hallucination_rate')):>7}"
            f" {m.get('n') or '--':>6}"
            f"{tag}"
        )
    print("(* = v2 基线)")


def print_group_b(rows: list[dict]):
    """Group B: 训练数据消融对比表"""
    print("\n" + "=" * 70)
    print("  Group B: 训练数据消融 (固定 v2 格式, 测试集: beam20_lam0.2)")
    print("=" * 70)
    header = f"{'Config':<20} {'Hit@1':>7} {'MacroF1':>8} {'EM':>7} {'CitAcc':>7} {'CitRec':>7} {'Hallu':>7} {'n':>6}"
    print(header)
    print("-" * 70)
    for r in rows:
        m = r.get("metrics") or {}
        tag = " *" if r.get("config") == "groupA_v2" else "  "
        print(
            f"{r.get('config','?'):<20}"
            f" {fmt_val(m.get('hit1')):>7}"
            f" {fmt_val(m.get('macro_f1')):>8}"
            f" {fmt_val(m.get('exact_match')):>7}"
            f" {fmt_val(m.get('citation_accuracy')):>7}"
            f" {fmt_val(m.get('citation_recall')):>7}"
            f" {fmt_val(m.get('hallucination_rate')):>7}"
            f" {m.get('n') or '--':>6}"
            f"{tag}"
        )
    print("(* = v2 基线，包含于 Group A)")


def print_group_c(rows: list[dict]):
    """Group C: 检索参数消融对比表"""
    print("\n" + "=" * 70)
    print("  Group C: 检索参数消融 (固定 v2 模型)")
    print("=" * 70)
    header = f"{'test_stem':<22} {'Hit@1':>7} {'MacroF1':>8} {'EM':>7} {'CitAcc':>7} {'CitRec':>7} {'n':>6}"
    print(header)
    print("-" * 70)
    # 按 test_stem 排序（先 beam 再 lam）
    for r in sorted(rows, key=lambda x: x.get("test_stem", "")):
        m = r.get("metrics") or {}
        tag = " *" if r.get("test_stem") == "beam20_lam0.2" else "  "
        print(
            f"{r.get('test_stem','?'):<22}"
            f" {fmt_val(m.get('hit1')):>7}"
            f" {fmt_val(m.get('macro_f1')):>8}"
            f" {fmt_val(m.get('exact_match')):>7}"
            f" {fmt_val(m.get('citation_accuracy')):>7}"
            f" {fmt_val(m.get('citation_recall')):>7}"
            f" {m.get('n') or '--':>6}"
            f"{tag}"
        )
    print("(* = 基线)")


# ─── CSV 写出 ─────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "group", "config", "test_stem", "fmt", "n",
    "hit1", "hit_any", "macro_p", "macro_r", "macro_f1",
    "micro_p", "micro_r", "micro_f1",
    "exact_match", "citation_accuracy", "citation_recall",
    "hallucination_rate", "format_compliance",
    "log_path",
]


def write_csv(all_rows: list[dict], csv_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in all_rows:
            m = r.get("metrics") or {}
            row = {
                "group":     r.get("group"),
                "config":    r.get("config"),
                "test_stem": r.get("test_stem"),
                "fmt":       r.get("fmt"),
                "log_path":  r.get("log_path"),
            }
            for key in ("n", "hit1", "hit_any", "macro_p", "macro_r", "macro_f1",
                        "micro_p", "micro_r", "micro_f1", "exact_match",
                        "citation_accuracy", "citation_recall",
                        "hallucination_rate", "format_compliance"):
                row[key] = m.get(key)
            writer.writerow(row)
    print(f"\n[INFO] CSV 已写出: {csv_path}")


# ─── 主函数 ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="消融实验结果汇总")
    p.add_argument(
        "--ablation_dir",
        default="data/output/WebQSP/ablation",
        help="消融实验输出目录（包含各 config 子目录）",
    )
    p.add_argument(
        "--baseline_log",
        default=None,
        help="已有基线 eval log 路径（可选，额外加入对比）"
             "例: data/output/WebQSP/grid_search/paths/beam20_lam0.2_v2_ft_eval.log",
    )
    p.add_argument(
        "--csv",
        default=None,
        help="CSV 输出路径（默认: {ablation_dir}/results.csv）",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # 解析为绝对路径（相对于脚本工作目录）
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    ablation_dir = str(project_dir / args.ablation_dir) if not os.path.isabs(args.ablation_dir) else args.ablation_dir
    csv_path = args.csv or os.path.join(ablation_dir, "results.csv")

    print(f"[INFO] 扫描目录: {ablation_dir}")

    records = discover_logs(ablation_dir)

    # 可选：加入外部基线 log
    if args.baseline_log and os.path.isfile(args.baseline_log):
        baseline_path = args.baseline_log
        stem = Path(baseline_path).stem  # beam20_lam0.2_v2_ft_eval
        inner = stem[: -len("_ft_eval")] if stem.endswith("_ft_eval") else stem
        parts = inner.rsplit("_", 1)
        test_stem, fmt = (parts[0], parts[1]) if len(parts) == 2 else (inner, "v2")
        records.append({
            "config":    "groupA_v2",
            "group":     "A",
            "log_path":  baseline_path,
            "test_stem": test_stem,
            "fmt":       fmt,
        })
        print(f"[INFO] 外部基线 log 已加入: {baseline_path}")

    if not records:
        print("[WARN] 未发现任何 eval log，请先运行 run_ablation.sh")
        sys.exit(0)

    # 解析所有日志
    for r in records:
        r["metrics"] = parse_eval_log(r["log_path"])
        status = f"n={r['metrics']['n']}" if r["metrics"] and r["metrics"]["n"] else "解析失败"
        print(f"  [{r['group']}] {r['config']:20s} {r['test_stem']:25s} {r['fmt']:4s} → {status}")

    # 分组
    group_a = [r for r in records if r["group"] == "A"]
    group_b = [r for r in records if r["group"] == "B"]
    group_c = [r for r in records if r["group"] == "C"]

    # 打印表格
    if group_a:
        # Group A 按 fmt 排序：v1 v2 v3 v4
        order = {"v1": 0, "v2": 1, "v3": 2, "v4": 3}
        group_a_sorted = sorted(group_a, key=lambda r: order.get(r.get("fmt", ""), 99))
        print_group_a(group_a_sorted)

    if group_b:
        print_group_b(group_b)

    if group_c:
        print_group_c(group_c)

    # 写 CSV
    write_csv(records, csv_path)


if __name__ == "__main__":
    main()
