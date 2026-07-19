"""汇总第三章多检索路径下游 QA 的已完成结果。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.ch3.downstream_qa import (
    CONDITION_IDS,
    extract_qa_metrics,
    summarize_input_paths,
)


def write_report(
    *,
    config: dict[str, Any],
    input_info: dict[str, dict[str, Any]],
    input_paths: dict[str, Path],
    layer_dir: Path,
    report_dir: Path,
) -> dict[str, Any]:
    """从各条件 summary.json 生成机器可读矩阵和中文 Markdown 摘要。"""
    rows: list[dict[str, Any]] = []
    labels = {item["id"]: item["label"] for item in config["conditions"]}
    methods = {item["id"]: item["method"] for item in config["conditions"]}
    for condition_id in CONDITION_IDS:
        summary_path = layer_dir / condition_id / "eval" / "summary.json"
        if not summary_path.is_file():
            raise ValueError(f"条件 {condition_id} 尚无完成的 QA 汇总: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append({
            "id": condition_id,
            "label": labels[condition_id],
            "method": methods[condition_id],
            "input": input_info[condition_id]["input"],
            "path": summarize_input_paths(
                input_paths[condition_id], no_paths=condition_id == "no_path"
            ),
            "qa": extract_qa_metrics(summary),
        })
    matrix = {
        "schema_version": 1,
        "dataset": config["dataset"],
        "backbone": config["backbone"],
        "config_id": config["config_id"],
        "profile": config["_profile_path"],
        "evaluation": config["evaluation"],
        "conditions": rows,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "condition_matrix.json").write_text(
        json.dumps(matrix, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    table = [
        "# 第三章多检索路径下游 QA 汇总",
        "",
        "本表只汇总固定模型、固定提示和确定性解码下的检索上下文对照；不与第四章训练源消融混写。",
        "",
        "## 上游路径质量（与本次 QA 使用相同输入）",
        "",
        "| 条件 | Path Answer Hit | Path Top1 Hit | Path P | Path R | Path Tail-Entity F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        path = row["path"]
        if path is None:
            table.append(f"| {row['label']} | 不适用 | 不适用 | 不适用 | 不适用 | 不适用 |")
            continue
        table.append(
            f"| {row['label']} | {path['answer_hit']:.4f} | {path['top1_hit']:.4f} | "
            f"{path['precision']:.4f} | {path['recall']:.4f} | {path['f1']:.4f} |"
        )
    table.extend([
        "",
        "## 上游路径多样性（与本次 QA 使用相同输入）",
        "",
        "| 条件 | 边 Jaccard 多样性 | 关系 Jaccard 多样性 | 尾节点多样性 | 关系覆盖率 | 边覆盖率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in rows:
        path = row["path"]
        if path is None:
            table.append(f"| {row['label']} | 不适用 | 不适用 | 不适用 | 不适用 | 不适用 |")
            continue
        table.append(
            f"| {row['label']} | {path['jaccard_diversity']:.4f} | "
            f"{path['relation_jaccard_diversity']:.4f} | {path['tail_diversity']:.4f} | "
            f"{path['relation_coverage']:.4f} | {path['edge_coverage']:.4f} |"
        )
    table.extend([
        "",
        "## 下游 QA 质量",
        "",
        "| 条件 | QA Hit@1 | QA Hit_any | QA Macro-F1 | QA Micro-F1 | EM |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in rows:
        qa = row["qa"]
        table.append(
            f"| {row['label']} | {qa['hit1']:.4f} | {qa['hit_any']:.4f} | "
            f"{qa['macro_f1']:.4f} | {qa['micro_f1']:.4f} | {qa['exact_match']:.4f} |"
        )
    (report_dir / "summary.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    return matrix
