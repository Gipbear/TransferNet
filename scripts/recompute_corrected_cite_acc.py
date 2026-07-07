"""离线重算 citation_accuracy / citation_recall(修正口径)。

旧实现把"已被精确性过滤(score_margin/hop_filter/topic 守卫)剔除出最终答案"的
引用路径仍计入 citation 分母,导致答案指标上升而 cite_acc 反常下降。修正口径用
`cited_indices_for_answers` 将引用集合对齐到最终答案集合后再算。

本脚本不重跑模型:直接读取各档 checked_batch_eval.jsonl 逐样本重算,再用修正值
覆盖原 summary 的 citation 字段,另存 *_corrected.json(保留原始 summary)。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oh_my_agent.common.metrics import (
    cited_indices_for_answers,
    compute_faithfulness,
    label_golden_indices,
)

RESULT_FILENAME = "checked_batch_eval.jsonl"
SUMMARY_FILENAME = "checked_batch_eval_summary.json"


def _recompute_dir(eval_dir: Path) -> tuple[float, float] | None:
    records_path = eval_dir / RESULT_FILENAME
    if not records_path.exists():
        return None
    cite_sum = 0.0
    recall_sum = 0.0
    n = 0
    with records_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            n += 1
            raw_paths = record["raw_mmr_reason_paths"]
            cited = set(record.get("final_accepted_path_indices", [])) | set(
                record.get("relation_expanded_path_indices", [])
            )
            aligned = cited_indices_for_answers(
                cited, raw_paths, record.get("pred_answer_disambiguated_mids", [])
            )
            golden = label_golden_indices(raw_paths, record["gold_mids"])
            faith = compute_faithfulness(aligned, golden, [], set())
            cite_sum += faith["citation_accuracy"]
            recall_sum += faith["citation_recall"]
    if n == 0:
        return None
    return round(cite_sum / n, 4), round(recall_sum / n, 4)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots", nargs="+", help="包含 checked_batch_eval.jsonl 的评测目录(可多个或其父目录)"
    )
    parser.add_argument(
        "--write", action="store_true", help="另存 *_corrected.json(否则仅打印对照)"
    )
    args = parser.parse_args(argv)

    dirs: list[Path] = []
    for root in args.roots:
        root_path = Path(root)
        if (root_path / RESULT_FILENAME).exists():
            dirs.append(root_path)
        else:
            dirs.extend(sorted(p.parent for p in root_path.rglob(RESULT_FILENAME)))

    print(f"{'目录':<46}{'旧 cite':>10}{'修正 cite':>12}")
    for eval_dir in dirs:
        result = _recompute_dir(eval_dir)
        if result is None:
            continue
        cite_corrected, recall_corrected = result
        summary_path = eval_dir / SUMMARY_FILENAME
        old_cite = None
        summary = {}
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            old_cite = summary.get("citation_accuracy")
        rel = eval_dir.relative_to(ROOT) if eval_dir.is_absolute() else eval_dir
        old_str = f"{old_cite:.4f}" if isinstance(old_cite, (int, float)) else "—"
        print(f"{str(rel):<46}{old_str:>10}{cite_corrected:>12.4f}")
        if args.write and summary:
            summary["citation_accuracy"] = cite_corrected
            summary["citation_recall"] = recall_corrected
            summary["citation_accuracy_definition"] = "aligned_to_final_answers"
            out_path = eval_dir / "checked_batch_eval_summary_corrected.json"
            out_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
