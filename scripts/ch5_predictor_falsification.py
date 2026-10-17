"""A2 事前判据的系统性否证：候选信号能否预测"补全是否带来正收益"。

对每个**不含 gold、可在部署时计算**的信号，在两数据集上分别计算
判别 AUC（触发样本的"有正收益 vs 无收益"），再做跨数据集符号一致性检验：
方向相反直接判不可用；方向一致但两侧判别力都不足也判不可用。

同时列出需要 gold 的量作为对照，说明"唯一有判别力的量为何不能当判据"。

复现：
    python -m scripts.ch5_predictor_falsification --output <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.ch5_applicability_probe import auc
from scripts.ch5_analysis_common import load_probe

# 事前可算信号：部署时不需要 gold 即可得到。
DEPLOYABLE: dict[str, list[str]] = {
    "路径结构": ["closed_n"],
    "模型一致性": ["group_largest", "group_union", "group_over_ans", "group_in_ratio"],
    "关系名词法": ["has_reverse", "seq_len", "any_reverse"],
    "检索信号": ["pred_size", "expanded_score_mean", "expanded_score_min", "expanded_in_pred"],
    "补全规模（可预计算）": ["expanded_n"],
}

# 需要 gold 的量：只能做事后归因，不能做部署判据。
GOLD_DEPENDENT = ["gold_n", "recall_closed"]

CN_LABELS = {
    "closed_n": "补全前答案数",
    "group_largest": "关系组最大尾实体数",
    "group_union": "关系组尾实体并集大小",
    "group_over_ans": "关系组规模 / 补全前答案数",
    "group_in_ratio": "答案落在关系组内的比例",
    "has_reverse": "主导关系组含反向关系",
    "seq_len": "主导关系组序列长度",
    "any_reverse": "任一组含反向关系",
    "pred_size": "检索候选池规模",
    "expanded_score_mean": "补进实体平均检索分",
    "expanded_score_min": "补进实体最低检索分",
    "expanded_in_pred": "补进实体在候选池中的占比",
    "expanded_n": "预计补进实体数",
    "gold_n": "答案规模（需 gold）",
    "recall_closed": "补全前召回（需 gold）",
}


def signal_aucs(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, dict[str, float]]:
    triggered = [row for row in rows if row["triggered"] == 1]
    positive = [row for row in triggered if row["delta_f1"] > 1e-9]
    negative = [row for row in triggered if row["delta_f1"] <= 1e-9]
    result = {}
    for key in keys:
        score = auc([float(row.get(key, 0.0)) for row in positive],
                    [float(row.get(key, 0.0)) for row in negative])
        result[key] = {"auc": score, "n_pos": len(positive), "n_neg": len(negative)}
    return result


def verdict(webqsp_auc: float, cwq_auc: float, min_strength: float = 0.10) -> str:
    """跨数据集判定：方向须一致，且两侧判别力都不能太弱。"""
    same_side = (webqsp_auc - 0.5) * (cwq_auc - 0.5) > 0
    strength = min(abs(webqsp_auc - 0.5), abs(cwq_auc - 0.5))
    if not same_side:
        return "**不可用**（方向相反）"
    if strength < min_strength:
        return "**不可用**（判别力不足）"
    return "候选"


def render_table(title: str, per_dataset: dict[str, dict[str, dict[str, float]]], keys: list[str]) -> str:
    lines = [f"### {title}", "",
             "| 信号 | WebQSP AUC | CWQ AUC | 判定 |", "| --- | ---: | ---: | --- |"]
    for key in keys:
        left = per_dataset["webqsp"][key]["auc"]
        right = per_dataset["cwq"][key]["auc"]
        lines.append(f"| {CN_LABELS.get(key, key)} (`{key}`) | {left:.3f} | {right:.3f} | "
                     f"{verdict(left, right)} |")
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
    deployable = sorted({key for group in DEPLOYABLE.values() for key in group})
    per_dataset = {name: signal_aucs(rows, deployable + GOLD_DEPENDENT)
                   for name, rows in datasets.items()}

    lines = [
        "# A2 事前判据的系统性否证", "",
        "判定口径：触发样本按「补全是否带来正收益（ΔF1 > 0）」二分，计算各信号的判别 AUC。",
        "AUC = 0.5 表示无判别力；跨数据集判定要求方向一致且两侧 |AUC−0.5| ≥ 0.10。",
        f"触发样本量：WebQSP {per_dataset['webqsp']['gold_n']['n_pos'] + per_dataset['webqsp']['gold_n']['n_neg']}"
        f"（正/无收益 {per_dataset['webqsp']['gold_n']['n_pos']}/{per_dataset['webqsp']['gold_n']['n_neg']}）、"
        f"CWQ {per_dataset['cwq']['gold_n']['n_pos'] + per_dataset['cwq']['gold_n']['n_neg']}"
        f"（正/无收益 {per_dataset['cwq']['gold_n']['n_pos']}/{per_dataset['cwq']['gold_n']['n_neg']}）。",
        "", "## 部署时可算的信号", "",
    ]
    for group, keys in DEPLOYABLE.items():
        lines.append(render_table(f"{group}", per_dataset, keys))
    lines.append("## 需要 gold 的量（对照，不可作部署判据）")
    lines.append("")
    lines.append(render_table("事后量", per_dataset, GOLD_DEPENDENT))

    candidates = [key for key in deployable
                  if verdict(per_dataset["webqsp"][key]["auc"], per_dataset["cwq"][key]["auc"]) == "候选"]
    lines.append("## 结论")
    lines.append("")
    if candidates:
        lines.append(f"通过跨数据集检验的候选信号：{', '.join(f'`{key}`' for key in candidates)}。")
    else:
        lines.append("**没有任何部署时可算的信号通过跨数据集检验**——方向相反或判别力不足。")
        lines.append("")
        lines.append("这与 A1 的机理结论一致：决定增益的关系组纯度是「问题 × 关系」的语义匹配性质，")
        lines.append("不反映在模型输出结构、关系名或检索分数里。判据必须来自问题级的语义判断（见路径 B）。")

    payload = {"per_dataset": per_dataset, "candidates": candidates,
               "deployable": deployable, "gold_dependent": GOLD_DEPENDENT}
    (args.output / "predictor_falsification.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = "\n".join(lines)
    (args.output / "predictor_falsification.md").write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
