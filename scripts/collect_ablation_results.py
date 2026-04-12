"""
消融实验结果汇总脚本

扫描 data/output/WebQSP/ablation/ 下所有 *_eval.log 文件，
解析关键指标，输出分组对比表格（CSV + 终端格式化），并生成可视化图表。

用法：
  python scripts/collect_ablation_results.py
  python scripts/collect_ablation_results.py --ablation_dir data/output/WebQSP/ablation
  python scripts/collect_ablation_results.py \\
      --baseline_log data/output/WebQSP/grid_search/paths/beam20_lam0.2_v2_ft_eval.log
  python scripts/collect_ablation_results.py --no_plot   # 跳过图表生成
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path


# ─── 绘图字体（可选依赖） ─────────────────────────────────────────────────────────

def _get_font(size: int = 12):
    """尝试加载微软雅黑，失败则返回 None（使用 matplotlib 默认字体）。"""
    try:
        from matplotlib.font_manager import FontProperties
        font_path = "/mnt/c/Windows/Fonts/msyh.ttc"
        if os.path.isfile(font_path):
            return FontProperties(fname=font_path, size=size)
    except Exception:
        pass
    return None


def _cn(text_cn: str, text_en: str, font) -> str:
    """字体存在时返回中文，否则返回英文。"""
    return text_cn if font is not None else text_en


# ─── 指标解析 ──────────────────────────────────────────────────────────────────

# 日志格式参照 eval_faithfulness.py:log_metrics() / log_metrics_with_std()
# _VS 匹配 "0.7644" 或 "0.7644 ± 0.0032" 或 "0.7644±0.0032"
_VS = r"([\d.]+)(?:\s*±\s*([\d.]+))?"

_METRIC_PATTERNS = {
    "hit1":               re.compile(rf"Hit@1\s*:\s*{_VS}"),
    "hit_any":            re.compile(rf"Hit@Any\s*:\s*{_VS}"),
    "macro_prf":          re.compile(rf"Macro\s+P/R/F1\s*:\s*{_VS}\s*/\s*{_VS}\s*/\s*{_VS}"),
    "micro_prf":          re.compile(rf"Micro\s+P/R/F1\s*:\s*{_VS}\s*/\s*{_VS}\s*/\s*{_VS}"),
    "exact_match":        re.compile(rf"Exact Match\s*:\s*{_VS}"),
    "citation_accuracy":  re.compile(rf"Citation Accuracy\s*:\s*{_VS}"),
    "citation_recall":    re.compile(rf"Citation Recall\s*:\s*{_VS}"),
    "hallucination_rate":  re.compile(rf"Hallucination Rate\s*:\s*{_VS}"),
    "format_compliance":   re.compile(rf"Format Compliance\s*:\s*{_VS}"),
    # 拒答指标（Group F）
    "rejection_precision": re.compile(rf"Rejection Precision\s*:\s*{_VS}"),
    "rejection_recall":    re.compile(rf"Rejection Recall\s*:\s*{_VS}"),
    "rejection_f1":        re.compile(rf"Rejection F1\s*:\s*{_VS}"),
    # 兼容 "[ ALL ... ]  (n=1541)" 和 "[ 多轮汇总 ... ]  (avg n=1541)"
    "n":                   re.compile(r"\[ (?:ALL|多轮汇总).*?\]\s+\((?:avg )?n=(\d+)\)"),
}


def parse_eval_log(log_path: str) -> dict | None:
    """
    解析 eval_faithfulness.py 生成的日志文件。

    优先从多轮汇总段（num_runs > 1）解析 mean ± std；
    若无多轮汇总段则回落到 [ ALL ... ] 段（单次推理，向后兼容）。
    返回指标 dict（含可选 *_std 字段），失败返回 None。
    """
    try:
        text = Path(log_path).read_text(encoding="utf-8")
    except OSError:
        return None

    # 优先解析多轮汇总段
    multi_start = text.find("[ 多轮汇总")
    if multi_start != -1:
        next_sep = text.find("---", multi_start)
        segment = text[multi_start:next_sep] if next_sep != -1 else text[multi_start:]
    else:
        # 回落：单次推理的第一个 [ ALL ... ] 块
        all_start = text.find("[ ALL")
        if all_start == -1:
            return None
        next_sep = text.find("---", all_start)
        segment = text[all_start:next_sep] if next_sep != -1 else text[all_start:]

    result = {}

    m = _METRIC_PATTERNS["n"].search(segment)
    result["n"] = int(m.group(1)) if m else None

    for key in ("hit1", "hit_any", "exact_match",
                "citation_accuracy", "citation_recall",
                "hallucination_rate", "format_compliance"):
        m = _METRIC_PATTERNS[key].search(segment)
        if m:
            result[key]            = float(m.group(1))
            result[f"{key}_std"]   = float(m.group(2)) if m.group(2) else None
        else:
            result[key]            = None
            result[f"{key}_std"]   = None

    m = _METRIC_PATTERNS["macro_prf"].search(segment)
    if m:
        result["macro_p"],     result["macro_p_std"]   = float(m.group(1)), (float(m.group(2)) if m.group(2) else None)
        result["macro_r"],     result["macro_r_std"]   = float(m.group(3)), (float(m.group(4)) if m.group(4) else None)
        result["macro_f1"],    result["macro_f1_std"]  = float(m.group(5)), (float(m.group(6)) if m.group(6) else None)
    else:
        result["macro_p"] = result["macro_r"] = result["macro_f1"] = None
        result["macro_p_std"] = result["macro_r_std"] = result["macro_f1_std"] = None

    m = _METRIC_PATTERNS["micro_prf"].search(segment)
    if m:
        result["micro_p"],     result["micro_p_std"]   = float(m.group(1)), (float(m.group(2)) if m.group(2) else None)
        result["micro_r"],     result["micro_r_std"]   = float(m.group(3)), (float(m.group(4)) if m.group(4) else None)
        result["micro_f1"],    result["micro_f1_std"]  = float(m.group(5)), (float(m.group(6)) if m.group(6) else None)
    else:
        result["micro_p"] = result["micro_r"] = result["micro_f1"] = None
        result["micro_p_std"] = result["micro_r_std"] = result["micro_f1_std"] = None

    # 拒答指标（Group F）：从 "--- Rejection Analysis ---" 段落解析
    rej_start = text.find("--- Rejection Analysis ---")
    if rej_start != -1:
        rej_end = text.find("---", rej_start + len("--- Rejection Analysis ---"))
        rej_segment = text[rej_start:rej_end] if rej_end != -1 else text[rej_start:]
        for key in ("rejection_precision", "rejection_recall", "rejection_f1"):
            m = _METRIC_PATTERNS[key].search(rej_segment)
            if m:
                result[key]          = float(m.group(1))
                result[f"{key}_std"] = float(m.group(2)) if m.group(2) else None
            else:
                result[key]          = None
                result[f"{key}_std"] = None
    else:
        for key in ("rejection_precision", "rejection_recall", "rejection_f1"):
            result[key]          = None
            result[f"{key}_std"] = None

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
    if config_name.startswith("groupF"):
        return "F"
    if config_name.startswith("groupI"):
        return "I"
    return "X"


# ─── 表格格式化 ────────────────────────────────────────────────────────────────

def fmt_val(v, std=None, precision: int = 4) -> str:
    """格式化指标值。std 不为 None 时输出 'v±std'。"""
    if v is None:
        return "--"
    if std is not None:
        return f"{v:.{precision}f}±{std:.{precision}f}"
    return f"{v:.{precision}f}"


def _has_std(rows: list[dict]) -> bool:
    """判断任意一行是否含有 std 数据（用于动态调整列宽）。"""
    return any((r.get("metrics") or {}).get("hit1_std") is not None for r in rows)


def print_group_a(rows: list[dict]):
    """Group A: 输出格式消融对比表"""
    std = _has_std(rows)
    cw = 14 if std else 7   # 指标列宽（含 ±std 时需更宽）
    total = 10 + 1 + (cw + 1) * 6 + 8
    print("\n" + "=" * total)
    print("  Group A: 输出格式消融 (测试集: beam20_lam0.2)")
    print("=" * total)
    header = (f"{'Format':<10}"
              f" {'Hit@1':>{cw}}"
              f" {'MacroF1':>{cw+1}}"
              f" {'EM':>{cw}}"
              f" {'CitAcc':>{cw}}"
              f" {'CitRec':>{cw}}"
              f" {'Hallu':>{cw}}"
              f" {'n':>6}")
    print(header)
    print("-" * total)
    for r in rows:
        m = r.get("metrics") or {}
        is_v1 = r.get("fmt") == "v1"
        cit_acc = "--" if is_v1 else fmt_val(m.get("citation_accuracy"), m.get("citation_accuracy_std"))
        cit_rec = "--" if is_v1 else fmt_val(m.get("citation_recall"),   m.get("citation_recall_std"))
        tag = " *" if r.get("config") == "groupA_v2" else "  "
        print(
            f"{r.get('fmt','?'):<10}"
            f" {fmt_val(m.get('hit1'),             m.get('hit1_std')):>{cw}}"
            f" {fmt_val(m.get('macro_f1'),          m.get('macro_f1_std')):>{cw+1}}"
            f" {fmt_val(m.get('exact_match'),       m.get('exact_match_std')):>{cw}}"
            f" {cit_acc:>{cw}}"
            f" {cit_rec:>{cw}}"
            f" {fmt_val(m.get('hallucination_rate'),m.get('hallucination_rate_std')):>{cw}}"
            f" {m.get('n') or '--':>6}"
            f"{tag}"
        )
    print("(* = v2 基线)" + ("  [±std = 多轮汇总]" if std else ""))


def print_group_b(rows: list[dict]):
    """Group B: 训练数据消融对比表"""
    std = _has_std(rows)
    cw = 14 if std else 7
    total = 20 + 1 + (cw + 1) * 6 + 8
    print("\n" + "=" * total)
    print("  Group B: 训练数据消融 (固定 v2 格式, 测试集: beam20_lam0.2)")
    print("=" * total)
    header = (f"{'Config':<20}"
              f" {'Hit@1':>{cw}}"
              f" {'MacroF1':>{cw+1}}"
              f" {'EM':>{cw}}"
              f" {'CitAcc':>{cw}}"
              f" {'CitRec':>{cw}}"
              f" {'Hallu':>{cw}}"
              f" {'n':>6}")
    print(header)
    print("-" * total)
    for r in rows:
        m = r.get("metrics") or {}
        tag = " *" if r.get("config") == "groupA_v2" else "  "
        print(
            f"{r.get('config','?'):<20}"
            f" {fmt_val(m.get('hit1'),             m.get('hit1_std')):>{cw}}"
            f" {fmt_val(m.get('macro_f1'),          m.get('macro_f1_std')):>{cw+1}}"
            f" {fmt_val(m.get('exact_match'),       m.get('exact_match_std')):>{cw}}"
            f" {fmt_val(m.get('citation_accuracy'), m.get('citation_accuracy_std')):>{cw}}"
            f" {fmt_val(m.get('citation_recall'),   m.get('citation_recall_std')):>{cw}}"
            f" {fmt_val(m.get('hallucination_rate'),m.get('hallucination_rate_std')):>{cw}}"
            f" {m.get('n') or '--':>6}"
            f"{tag}"
        )
    print("(* = v2 基线，包含于 Group A)" + ("  [±std = 多轮汇总]" if std else ""))


def print_group_c(rows: list[dict]):
    """Group C: 检索参数消融对比表"""
    std = _has_std(rows)
    cw = 14 if std else 7
    total = 22 + 1 + (cw + 1) * 5 + 8
    print("\n" + "=" * total)
    print("  Group C: 检索参数消融 (固定 v2 模型)")
    print("=" * total)
    header = (f"{'test_stem':<22}"
              f" {'Hit@1':>{cw}}"
              f" {'MacroF1':>{cw+1}}"
              f" {'EM':>{cw}}"
              f" {'CitAcc':>{cw}}"
              f" {'CitRec':>{cw}}"
              f" {'n':>6}")
    print(header)
    print("-" * total)
    for r in sorted(rows, key=lambda x: x.get("test_stem", "")):
        m = r.get("metrics") or {}
        tag = " *" if r.get("test_stem") == "beam20_lam0.2" else "  "
        print(
            f"{r.get('test_stem','?'):<22}"
            f" {fmt_val(m.get('hit1'),             m.get('hit1_std')):>{cw}}"
            f" {fmt_val(m.get('macro_f1'),          m.get('macro_f1_std')):>{cw+1}}"
            f" {fmt_val(m.get('exact_match'),       m.get('exact_match_std')):>{cw}}"
            f" {fmt_val(m.get('citation_accuracy'), m.get('citation_accuracy_std')):>{cw}}"
            f" {fmt_val(m.get('citation_recall'),   m.get('citation_recall_std')):>{cw}}"
            f" {m.get('n') or '--':>6}"
            f"{tag}"
        )
    print("(* = 基线)" + ("  [±std = 多轮汇总]" if std else ""))


def print_group_f(rows: list[dict]):
    """Group F: 拒答能力消融对比表"""
    std = _has_std(rows)
    cw = 14 if std else 7
    total = 20 + 1 + (cw + 1) * 7 + 8
    print("\n" + "=" * total)
    print("  Group F: 拒答能力训练 (chain+v2, MID vs Name)")
    print("=" * total)
    header = (f"{'Config':<20}"
              f" {'Hit@1':>{cw}}"
              f" {'MacroF1':>{cw+1}}"
              f" {'EM':>{cw}}"
              f" {'Rej.Prec':>{cw}}"
              f" {'Rej.Rec':>{cw}}"
              f" {'Rej.F1':>{cw}}"
              f" {'n':>6}")
    print(header)
    print("-" * total)
    for r in sorted(rows, key=lambda x: x.get("config", "")):
        m = r.get("metrics") or {}
        print(
            f"{r.get('config','?'):<20}"
            f" {fmt_val(m.get('hit1'),                m.get('hit1_std')):>{cw}}"
            f" {fmt_val(m.get('macro_f1'),             m.get('macro_f1_std')):>{cw+1}}"
            f" {fmt_val(m.get('exact_match'),          m.get('exact_match_std')):>{cw}}"
            f" {fmt_val(m.get('rejection_precision'),  m.get('rejection_precision_std')):>{cw}}"
            f" {fmt_val(m.get('rejection_recall'),     m.get('rejection_recall_std')):>{cw}}"
            f" {fmt_val(m.get('rejection_f1'),         m.get('rejection_f1_std')):>{cw}}"
            f" {m.get('n') or '--':>6}"
        )
    print("  Hit@1/MacroF1/EM 仅统计作答样本，Rej.* 为拒答 P/R/F1"
          + ("  [±std = 多轮汇总]" if std else ""))


def print_group_i(rows: list[dict]):
    """Group I: 问题特殊 token 对照表 (rawq vs stripq)"""
    std = _has_std(rows)
    cw = 14 if std else 7
    total = 38 + 1 + (cw + 1) * 5 + 8
    print("\n" + "=" * total)
    print("  Group I: 问题特殊 token 消融 (chain+v2+name)")
    print("=" * total)
    header = (f"{'Config':<38}"
              f" {'Hit@1':>{cw}}"
              f" {'MacroF1':>{cw+1}}"
              f" {'EM':>{cw}}"
              f" {'CitAcc':>{cw}}"
              f" {'CitRec':>{cw}}"
              f" {'n':>6}")
    print(header)
    print("-" * total)
    for r in sorted(rows, key=lambda x: x.get("config", "")):
        m = r.get("metrics") or {}
        tag = " *" if r.get("config") == "groupI_chain_v2_name_stripq" else "  "
        print(
            f"{r.get('config', '?'):<38}"
            f" {fmt_val(m.get('hit1'),             m.get('hit1_std')):>{cw}}"
            f" {fmt_val(m.get('macro_f1'),          m.get('macro_f1_std')):>{cw+1}}"
            f" {fmt_val(m.get('exact_match'),       m.get('exact_match_std')):>{cw}}"
            f" {fmt_val(m.get('citation_accuracy'), m.get('citation_accuracy_std')):>{cw}}"
            f" {fmt_val(m.get('citation_recall'),   m.get('citation_recall_std')):>{cw}}"
            f" {m.get('n') or '--':>6}"
            f"{tag}"
        )
    print("(* = stripq，去除 [CLS]/[SEP]/## 的对照组)"
          + ("  [±std = 多轮汇总]" if std else ""))


# ─── CSV 写出 ─────────────────────────────────────────────────────────────────

_METRIC_KEYS = [
    "hit1", "hit_any", "macro_p", "macro_r", "macro_f1",
    "micro_p", "micro_r", "micro_f1",
    "exact_match", "citation_accuracy", "citation_recall",
    "hallucination_rate", "format_compliance",
    "rejection_precision", "rejection_recall", "rejection_f1",
]

CSV_FIELDS = (
    ["group", "config", "test_stem", "fmt", "n"]
    + _METRIC_KEYS
    + [f"{k}_std" for k in _METRIC_KEYS]
    + ["log_path"]
)


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
                "n":         m.get("n"),
            }
            for key in _METRIC_KEYS:
                row[key]            = m.get(key)
                row[f"{key}_std"]   = m.get(f"{key}_std")
            writer.writerow(row)
    print(f"\n[INFO] CSV 已写出: {csv_path}")


# ─── 可视化 ──────────────────────────────────────────────────────────────────

def _save_fig(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[INFO] 图表已保存: {path}")


def plot_group_a(rows: list[dict], out_dir: str):
    """Group A: 输出格式消融 — 分组柱状图（Hit@1 / Macro F1 / EM / Citation Accuracy）"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[WARN] matplotlib 未安装，跳过图表生成")
        return

    font = _get_font(12)
    order = {"v1": 0, "v2": 1, "v3": 2, "v4": 3}
    rows = sorted(rows, key=lambda r: order.get(r.get("fmt", ""), 99))

    labels = [r.get("fmt", "?") for r in rows]
    metrics_def = [
        ("hit1",             "Hit@1",    "#4e79a7"),
        ("macro_f1",         "Macro F1", "#f28e2b"),
        ("exact_match",      "EM",       "#59a14f"),
        ("citation_accuracy","Cit.Acc",  "#e15759"),
    ]

    n_groups = len(labels)
    n_metrics = len(metrics_def)
    x = np.arange(n_groups)
    width = 0.18
    offsets = np.linspace(-(n_metrics - 1) / 2, (n_metrics - 1) / 2, n_metrics) * width

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.rcParams.update({"font.size": 12})

    for i, (key, label, color) in enumerate(metrics_def):
        vals, errs = [], []
        for r in rows:
            m = r.get("metrics") or {}
            if key == "citation_accuracy" and r.get("fmt") == "v1":
                vals.append(float("nan"))
                errs.append(0.0)
            else:
                vals.append(m.get(key) if m.get(key) is not None else float("nan"))
                errs.append(m.get(f"{key}_std") or 0.0)
        yerr = errs if any(e > 0 for e in errs) else None
        bars = ax.bar(x + offsets[i], vals, width, label=label, color=color, alpha=0.85,
                      yerr=yerr, capsize=3, error_kw={"elinewidth": 1.2, "alpha": 0.8})
        # 在柱顶显示数值
        for bar, v in zip(bars, vals):
            if not (v != v):  # not NaN
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    # 高亮 v2 基线列（加虚线框）
    for r_idx, r in enumerate(rows):
        if r.get("fmt") == "v2":
            ax.axvspan(x[r_idx] - 0.45, x[r_idx] + 0.45,
                       color="gold", alpha=0.15, zorder=0, label="v2 基线" if font else "v2 baseline")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    title = _cn("Group A：输出格式消融 (beam20_lam0.2)", "Group A: Output Format Ablation (beam20_lam0.2)", font)
    if font:
        ax.set_title(title, fontsize=13, fontproperties=font)
    else:
        ax.set_title(title, fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)

    _save_fig(fig, os.path.join(out_dir, "figures", "group_a_format.png"))
    plt.close(fig)


def plot_group_b(rows: list[dict], baseline_row: dict | None, out_dir: str):
    """Group B: 训练数据消融 — 分组柱状图（含基线对比）"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[WARN] matplotlib 未安装，跳过图表生成")
        return

    font = _get_font(12)

    # 将基线插入首位
    display_rows = []
    if baseline_row:
        br = dict(baseline_row)
        br["_label"] = "baseline"
        display_rows.append(br)
    config_labels = {
        "groupB_noshuffle": "no_shuffle",
        "groupB_noscore":   "no_score",
        "groupB_dist0.3":   "dist_0.3",
        "groupB_dist0.5":   "dist_0.5",
    }
    config_order = list(config_labels.keys())
    for cfg in config_order:
        for r in rows:
            if r.get("config") == cfg:
                rd = dict(r)
                rd["_label"] = config_labels[cfg]
                display_rows.append(rd)

    if not display_rows:
        print("[INFO] Group B 无数据，跳过图表")
        return

    labels = [r["_label"] for r in display_rows]
    metrics_def = [
        ("hit1",             "Hit@1",    "#4e79a7"),
        ("macro_f1",         "Macro F1", "#f28e2b"),
        ("exact_match",      "EM",       "#59a14f"),
        ("citation_accuracy","Cit.Acc",  "#e15759"),
    ]

    n_groups = len(labels)
    n_metrics = len(metrics_def)
    x = np.arange(n_groups)
    width = 0.18
    offsets = np.linspace(-(n_metrics - 1) / 2, (n_metrics - 1) / 2, n_metrics) * width

    fig, ax = plt.subplots(figsize=(max(10, n_groups * 2), 5))
    plt.rcParams.update({"font.size": 12})

    for i, (key, label, color) in enumerate(metrics_def):
        vals, errs = [], []
        for r in display_rows:
            m = r.get("metrics") or {}
            vals.append(m.get(key) if m.get(key) is not None else float("nan"))
            errs.append(m.get(f"{key}_std") or 0.0)
        yerr = errs if any(e > 0 for e in errs) else None
        bars = ax.bar(x + offsets[i], vals, width, label=label, color=color, alpha=0.85,
                      yerr=yerr, capsize=3, error_kw={"elinewidth": 1.2, "alpha": 0.8})
        for bar, v in zip(bars, vals):
            if not (v != v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    # 高亮 baseline 列
    if display_rows and display_rows[0]["_label"] == "baseline":
        ax.axvspan(x[0] - 0.45, x[0] + 0.45,
                   color="gold", alpha=0.15, zorder=0, label="baseline")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    title = _cn("Group B：训练数据消融 (v2 格式, beam20_lam0.2)", "Group B: Training Data Ablation (v2, beam20_lam0.2)", font)
    if font:
        ax.set_title(title, fontsize=13, fontproperties=font)
    else:
        ax.set_title(title, fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)

    _save_fig(fig, os.path.join(out_dir, "figures", "group_b_training.png"))
    plt.close(fig)


def _parse_beam_lam(test_stem: str):
    """从 'beam20_lam0.2' 解析出 (beam, lam)，解析失败返回 (None, None)。"""
    m = re.match(r"beam(\d+)_lam([\d.]+)", test_stem)
    if m:
        return int(m.group(1)), float(m.group(2))
    return None, None


def plot_group_c(rows: list[dict], out_dir: str):
    """Group C: 检索参数消融 — 2×2 子图折线图

    布局：
      行 0: beam_size 扫描 (λ=0.2)  |  左：答案指标   右：忠实度指标
      行 1: λ 扫描 (beam=20)         |  左：答案指标   右：忠实度指标

    答案指标：Hit@1, Macro F1, EM
    忠实度指标：Citation Accuracy, Citation Recall, Hallucination Rate (右轴)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib 未安装，跳过图表生成")
        return

    if not rows:
        print("[INFO] Group C 无数据，跳过图表")
        return

    font = _get_font(11)

    # 为每条记录补充 beam / lam
    for r in rows:
        b, l = _parse_beam_lam(r.get("test_stem", ""))
        r["_beam"] = b
        r["_lam"] = l

    # 固定 lambda=0.2，扫 beam
    beam_rows = sorted(
        [r for r in rows if r["_lam"] == 0.2 and r["_beam"] is not None],
        key=lambda r: r["_beam"]
    )
    # 固定 beam=20，扫 lambda
    lam_rows = sorted(
        [r for r in rows if r["_beam"] == 20 and r["_lam"] is not None],
        key=lambda r: r["_lam"]
    )

    if not beam_rows and not lam_rows:
        print("[INFO] Group C 无有效检索参数数据，跳过图表")
        return

    # 答案指标定义
    ans_metrics = [
        ("hit1",        "Hit@1",    "o", "#4e79a7"),
        ("macro_f1",    "Macro F1", "s", "#f28e2b"),
        ("exact_match", "EM",       "^", "#59a14f"),
    ]
    # 忠实度指标定义（Citation 用左轴，Hallucination 用右轴）
    faith_metrics_left = [
        ("citation_accuracy", "Cit.Acc",  "o", "#e15759"),
        ("citation_recall",   "Cit.Rec",  "s", "#76b7b2"),
    ]
    faith_hallu = ("hallucination_rate", "Hallu.Rate", "^", "#b07aa1")

    def _vals(rlist, x_key, metric_key):
        xs, ys, es = [], [], []
        for r in rlist:
            m = r.get("metrics") or {}
            xs.append(r[x_key])
            ys.append(m.get(metric_key))
            es.append(m.get(f"{metric_key}_std"))
        return xs, ys, es

    def _mark_baseline(ax, rlist, x_key, metric_key, color):
        """在基线点（beam20_lam0.2）上叠加 ★ 标记。"""
        for r in rlist:
            if r.get("_beam") == 20 and r.get("_lam") == 0.2:
                m = r.get("metrics") or {}
                v = m.get(metric_key)
                if v is not None:
                    ax.plot(r[x_key], v, marker="*", color=color,
                            markersize=13, zorder=6, linestyle="None")

    def _plot_line_with_band(ax, xs, ys, es, marker, color, linewidth, label, linestyle="-"):
        """绘制折线，若 es 中有 std 数据则叠加半透明误差带。"""
        import numpy as np
        ax.plot(xs, ys, marker=marker, color=color, linewidth=linewidth,
                label=label, linestyle=linestyle)
        if es and any(e is not None and e > 0 for e in es):
            ys_arr = np.array([y if y is not None else float("nan") for y in ys])
            es_arr = np.array([e if e is not None else 0.0 for e in es])
            ax.fill_between(xs, ys_arr - es_arr, ys_arr + es_arr,
                            alpha=0.15, color=color)

    def _draw_ans(ax, rlist, x_key, xticks, title_cn, title_en, xlabel):
        for key, label, marker, color in ans_metrics:
            xs, ys, es = _vals(rlist, x_key, key)
            _plot_line_with_band(ax, xs, ys, es, marker, color, 2, label)
            _mark_baseline(ax, rlist, x_key, key, color)
        ax.set_xticks(xticks)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_xlabel(xlabel, fontsize=11)
        t = title_cn if font else title_en
        ax.set_title(t, fontsize=12, fontproperties=font if font else None)
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(linestyle=":", alpha=0.6)
        ax.set_axisbelow(True)

    def _draw_faith(ax, rlist, x_key, xticks, title_cn, title_en, xlabel):
        # 左轴：Citation Accuracy, Citation Recall
        for key, label, marker, color in faith_metrics_left:
            xs, ys, es = _vals(rlist, x_key, key)
            _plot_line_with_band(ax, xs, ys, es, marker, color, 2, label)
            _mark_baseline(ax, rlist, x_key, key, color)
        ax.set_xticks(xticks)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Citation Score", fontsize=11)
        ax.set_xlabel(xlabel, fontsize=11)
        t = title_cn if font else title_en
        ax.set_title(t, fontsize=12, fontproperties=font if font else None)

        # 右轴：Hallucination Rate（通常很小，用右轴避免压缩左轴）
        ax2 = ax.twinx()
        key, label, marker, color = faith_hallu
        xs, ys, es = _vals(rlist, x_key, key)
        _plot_line_with_band(ax2, xs, ys, es, marker, color, 2, label, linestyle="--")
        _mark_baseline(ax2, rlist, x_key, key, color)
        ax2.set_ylabel("Hallucination Rate", fontsize=10, color=color)
        ax2.tick_params(axis="y", labelcolor=color)
        ax2.set_ylim(bottom=0)

        # 合并图例
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, fontsize=9, loc="upper right")
        ax.grid(linestyle=":", alpha=0.6)
        ax.set_axisbelow(True)

    # ── 确定行数（有数据的行才绘制）────────────────────────────────────────────
    n_rows = int(bool(beam_rows)) + int(bool(lam_rows))
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows))
    plt.rcParams.update({"font.size": 11})
    if n_rows == 1:
        axes = [axes]   # 保证 axes 始终是 2D list

    row_idx = 0
    if beam_rows:
        xticks = [r["_beam"] for r in beam_rows]
        xlabel = "Beam Size"
        _draw_ans(axes[row_idx][0], beam_rows, "_beam", xticks,
                  "答案指标 (λ=0.2)", "Answer Metrics (λ=0.2)", xlabel)
        _draw_faith(axes[row_idx][1], beam_rows, "_beam", xticks,
                    "忠实度指标 (λ=0.2)", "Faithfulness Metrics (λ=0.2)", xlabel)
        row_idx += 1

    if lam_rows:
        xticks = [r["_lam"] for r in lam_rows]
        xlabel = "λ (MMR Diversity Weight)"
        _draw_ans(axes[row_idx][0], lam_rows, "_lam", xticks,
                  "答案指标 (beam=20)", "Answer Metrics (beam=20)", xlabel)
        _draw_faith(axes[row_idx][1], lam_rows, "_lam", xticks,
                    "忠实度指标 (beam=20)", "Faithfulness Metrics (beam=20)", xlabel)

    # 全局标注
    fig.text(0.5, 0.005,
             "★ = beam20_lam0.2 基线" if font else "★ = beam20_lam0.2 baseline",
             ha="center", fontsize=10,
             fontproperties=font if font else None)

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    _save_fig(fig, os.path.join(out_dir, "figures", "group_c_retrieval.png"))
    plt.close(fig)


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
    p.add_argument(
        "--no_plot",
        action="store_true",
        help="跳过图表生成",
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
    group_f = [r for r in records if r["group"] == "F"]
    group_i = [r for r in records if r["group"] == "I"]

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

    if group_f:
        print_group_f(group_f)

    if group_i:
        print_group_i(group_i)

    # 写 CSV
    write_csv(records, csv_path)

    # ── 生成图表 ────────────────────────────────────────────────────────────────
    if not args.no_plot:
        figures_dir = ablation_dir
        # Group A
        if group_a:
            order = {"v1": 0, "v2": 1, "v3": 2, "v4": 3}
            plot_group_a(
                sorted(group_a, key=lambda r: order.get(r.get("fmt", ""), 99)),
                figures_dir,
            )
        # Group B — 找基线行（来自 Group A 的 v2 或外部 baseline_log）
        baseline_row = next(
            (r for r in group_a if r.get("fmt") == "v2"), None
        )
        if group_b or baseline_row:
            plot_group_b(group_b, baseline_row, figures_dir)
        # Group C
        if group_c:
            plot_group_c(group_c, figures_dir)


if __name__ == "__main__":
    main()
