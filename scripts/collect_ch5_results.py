"""汇总第五章实验对照表。

读取 run_ch5_experiments.sh 输出根目录下各子目录的 summary,并从 canonical 的
initial_answer.jsonl 聚合出"首答"基线,打印 Markdown 对照表。

用法:
    python scripts/collect_ch5_results.py <OUTPUT_ROOT>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kgqa.agent.common.metrics import aggregate_metrics

# 列:展示名 -> summary 键
COLUMNS = [
    ("n", "n"),
    ("hit1", "hit1"),
    ("hit_any", "hit_any"),
    ("macro_f1", "macro_f1"),
    ("micro_f1", "micro_f1"),
    ("EM", "exact_match"),
    ("cite_acc", "citation_accuracy"),
]

# 行顺序与中文标签(只展示存在的)
# 官方完整管线 = no_constrained(普通推理:check 自由解码 + 解析器兜底)。
# 消融阶梯亦从 no_constrained 回放,全程口径一致;canonical(受限解码)不作展示。
ROW_LABELS = [
    ("__initial__", "首次直接回答(baseline)"),
    ("ablation_base", "+ 自校验(无后处理)"),
    ("ablation_margin", "+ score_margin"),
    ("explore_best_score0p5_hopoff_top4_max3", "+ 关系补全 = 完整管线"),
    ("no_constrained", "+ topic 守卫 = 完整管线"),
    ("check_loose_only", "check: loose-only"),
    ("check_strict_only", "check: strict-only"),
    ("no_loopback", "loopback off(同环境对照)"),
]


def load_summary(d: Path) -> dict | None:
    f = d / "checked_batch_eval_summary.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def initial_answer_summary(root: Path) -> dict | None:
    """从 canonical(或任一可用)run 的 initial_answer.jsonl 聚合首答指标。"""
    for name in ("full_trace", "no_constrained", "ablation_base", "canonical"):
        f = root / name / "initial_answer.jsonl"
        if f.exists():
            recs = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
            if recs:
                return aggregate_metrics(recs)
    return None


def fmt(summary: dict, key: str) -> str:
    if key not in summary:
        return "-"
    v = summary[key]
    return str(v) if key == "n" else f"{v:.4f}"


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    root = Path(sys.argv[1])
    if not root.is_dir():
        print(f"目录不存在: {root}")
        return 1

    rows: list[tuple[str, dict]] = []
    for key, label in ROW_LABELS:
        summary = initial_answer_summary(root) if key == "__initial__" else load_summary(root / key)
        if summary:
            rows.append((label, summary))

    if not rows:
        print(f"{root} 下没有找到任何 summary,先运行实验。")
        return 1

    header = "| 配置 | " + " | ".join(name for name, _ in COLUMNS) + " |"
    sep = "|" + "---|" * (len(COLUMNS) + 1)
    print(f"\n# 第五章实验对照表  ({root.name})\n")
    print(header)
    print(sep)
    base = None
    for label, summary in rows:
        cells = [fmt(summary, key) for _, key in COLUMNS]
        # 相对完整管线给 EM 的 Δ 提示(可选,放在标签后)
        print(f"| {label} | " + " | ".join(cells) + " |")
        if label.endswith("完整管线"):
            base = summary
    print()
    if base:
        print(f"完整管线: macro_f1={base['macro_f1']:.4f}  EM={base['exact_match']:.4f}  "
              f"hit1={base['hit1']:.4f}  cite_acc={base['citation_accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
