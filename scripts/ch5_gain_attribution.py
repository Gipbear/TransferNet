"""A1 增益归因分解：把 ΔF1（开补全 − 关补全）分解到四个维度。

口径：F0（max_batches=2 + 原生早停），ΔF1 为逐样本配对差值。
区间为 2000 次逐样本配对 bootstrap 的 95% 分位区间（seed=17）。
维度：答案规模、补全前召回、关系组尾实体规模、关系组纯度。
另给「组纯度 × 补全前召回」二维交叉，用于分离机理（纯度）与操作量（漏答程度）。

复现：
    python -m scripts.ch5_gain_attribution --output <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from scripts.ch5_analysis_common import bootstrap_delta, bucketize, load_probe

DIMENSIONS: dict[str, list[float]] = {
    "gold_n": [2, 5, 20],
    "recall_closed": [0.25, 0.5, 0.75],
    "group_largest": [5, 20, 80],
    "expansion_precision": [0.25, 0.5, 0.75],
}

PURITY_BINS: tuple[tuple[str, Callable[[float], bool]], ...] = (
    ("纯度=0（补进全错）", lambda value: value <= 0.0),
    ("纯度∈(0,1)", lambda value: 0.0 < value < 1.0),
    ("纯度=1（补进全对）", lambda value: value >= 1.0),
)

RECALL_BINS: tuple[tuple[str, Callable[[float], bool]], ...] = (
    ("召回<0.25", lambda value: value < 0.25),
    ("召回[0.25,0.5)", lambda value: 0.25 <= value < 0.5),
    ("召回[0.5,0.75)", lambda value: 0.5 <= value < 0.75),
    ("召回≥0.75", lambda value: value >= 0.75),
)

LABELS = {
    "gold_n": "答案规模 gold_n",
    "recall_closed": "补全前召回 recall_closed",
    "group_largest": "关系组尾实体规模 group_largest",
    "expansion_precision": "关系组纯度 expansion_precision（仅触发样本）",
}


def attach_precision(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        if row["expanded_n"] > 0:
            row["expansion_precision"] = row["expanded_gold"] / row["expanded_n"]
        else:
            row["expansion_precision"] = 0.0


def expansion_precision(rows: list[dict[str, Any]]) -> float:
    usable = [row for row in rows if row["expanded_n"] > 0]
    if not usable:
        return 0.0
    return sum(row["expanded_gold"] / row["expanded_n"] for row in usable) / len(usable)


def format_ci(stat: dict[str, float], key: str) -> str:
    return (f"{stat[key] * 100:+.2f} [{stat[key + '_lo'] * 100:+.2f}, "
            f"{stat[key + '_hi'] * 100:+.2f}]")


def render_bucket_table(
    rows: list[dict[str, Any]], key: str, edges: list[float], subset: Callable | None = None
) -> str:
    lines = ["| 区间 | N | 触发 | ΔMacro-F1 [95%CI] | ΔMicro-F1 [95%CI] | 补全精度 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for label, members in bucketize(rows, key, edges):
        if subset is not None:
            members = [row for row in members if subset(row)]
        if members:
            stat = bootstrap_delta(members)
            triggered = sum(row["triggered"] for row in members)
            cells = (f"| {label} | {len(members)} | {triggered} | {format_ci(stat, 'macro')} | "
                     f"{format_ci(stat, 'micro')} | {expansion_precision(members) * 100:.1f}% |")
        else:
            cells = f"| {label} | 0 | 0 | — | — | — |"
        lines.append(cells)
    return "\n".join(lines)


def render_dimensions(rows: list[dict[str, Any]], title: str) -> str:
    lines = [f"### {title}", ""]
    for key, edges in DIMENSIONS.items():
        lines.append(f"**{LABELS[key]}**")
        lines.append("")
        subset = (lambda row: row["triggered"] == 1) if key == "expansion_precision" else None
        lines.append(render_bucket_table(rows, key, edges, subset))
        lines.append("")
    return "\n".join(lines)


def render_cross(
    rows: list[dict[str, Any]],
    title: str,
    row_name: str,
    row_bins: tuple[tuple[str, Any], ...],
    col_name: str,
    col_bins: tuple[tuple[str, Any], ...],
) -> str:
    """二维交叉表：控制 row 维度后看 col 维度的效应，单元格为 ΔMicro-F1 / N（触发样本）。

    分档用固定语义边界而非分位数——纯度在触发样本上高度集中在 1.0，
    分位数会把 82/100 个样本压进同一档，失去区分度。
    """
    triggered = [row for row in rows if row["triggered"] == 1]
    for position, row in enumerate(triggered):
        row["_uid"] = position

    def split(key: str, bins: tuple[tuple[str, Any], ...]) -> list[tuple[str, list[dict]]]:
        return [(label, [row for row in triggered if predicate(row[key])])
                for label, predicate in bins]

    row_groups, col_groups = split(row_name, row_bins), split(col_name, col_bins)
    header = f"| {row_name} \\ {col_name} | " + " | ".join(label for label, _ in col_groups) + " |"
    lines = [f"**{title}**（单元格：ΔMicro-F1 / N）", "", header,
             "| --- " + "| ---: " * len(col_groups) + "|"]
    for row_label, row_members in row_groups:
        keys = {member["_uid"] for member in row_members}
        cells = []
        for _, col_members in col_groups:
            cell = [member for member in col_members if member["_uid"] in keys]
            if cell:
                stat = bootstrap_delta(cell)
                cells.append(f"{stat['micro'] * 100:+.2f} / {len(cell)}")
            else:
                cells.append("— / 0")
        lines.append(f"| {row_label} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probe", type=Path,
                        default=Path("data/analysis/20260919_1700__ch5_applicability_probe"
                                     "/probe_results.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    datasets = load_probe(args.probe)
    for rows in datasets.values():
        attach_precision(rows)

    payload: dict[str, Any] = {}
    lines = [
        "# A1 增益归因分解", "",
        "口径：ΔF1 = F1(开补全) − F1(关补全)，F0（max_batches=2 + 原生早停）。",
        "区间为 2000 次逐样本配对 bootstrap 的 95% 分位区间（seed=17）。",
        "「补全精度」= 补进实体中属于 gold 的比例，仅触发样本有定义。",
        "两处读表提示：桶内样本数为 1 时区间宽度必然为 0（重采样只能抽到同一样本），",
        "不代表效应确定；「关系组纯度」分桶只覆盖触发样本，未触发样本不在该表内。", "",
        "## 总体", "",
        "| 数据集 | N | 触发 | ΔMacro-F1 [95%CI] | ΔMicro-F1 [95%CI] |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, rows in datasets.items():
        stat = bootstrap_delta(rows)
        payload[name] = {"overall": stat}
        lines.append(f"| {name} | {stat['n']} | {sum(r['triggered'] for r in rows)} | "
                     f"{format_ci(stat, 'macro')} | {format_ci(stat, 'micro')} |")
    lines.append("")

    for name, rows in datasets.items():
        lines.append(render_dimensions(rows, f"{name}：四维度分桶"))
        payload[name]["dimensions"] = {
            key: [{"range": label, "n": len(members),
                   "delta": bootstrap_delta(members) if members else None}
                  for label, members in bucketize(rows, key, edges)]
            for key, edges in DIMENSIONS.items()
        }
    lines.append("## 交叉控制（触发样本）")
    lines.append("")
    lines.append("控制「补全前召回」后看「组纯度」的效应，以及反向控制；")
    lines.append("用于判断增益究竟由图谱侧纯度决定，还是由模型侧漏答程度决定。")
    lines.append("")
    for name, rows in datasets.items():
        lines.append(render_cross(rows, name, "recall_closed", RECALL_BINS,
                                  "expansion_precision", PURITY_BINS))
        lines.append(render_cross(rows, name, "expansion_precision", PURITY_BINS,
                                  "recall_closed", RECALL_BINS))

    (args.output / "gain_attribution.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = "\n".join(lines)
    (args.output / "gain_attribution.md").write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
