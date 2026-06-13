"""从 canonical 录制离线回放第五章后处理消融阶梯,不重跑 LLM。

base / +margin / +hop / +expansion 与 canonical 的检索、答题、check **完全相同**,
只差末端确定性后处理。本脚本把 canonical 的 ``checked_batch_eval.jsonl`` 用 mock 工具
喂回**真实** agent,逐档复现后处理并写出与真实评测同结构的 summary,秒级完成。

用法:
    python scripts/replay_ch5_ablation.py \
        --canonical_dir <OUTPUT_ROOT>/canonical \
        --output_root   <OUTPUT_ROOT>

校验:脚本会先用 canonical 自身配置回放一遍,与录制 summary 的关键指标比对,
确认回放在真实数据上忠实复现(不一致即报错,不写出任何档位)。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oh_my_agent.agent.checked_batch_replay import replay_record
from oh_my_agent.common import (
    build_eval_record,
    compute_answer_metrics,
    compute_faithfulness,
    get_all_path_entities,
    label_golden_indices,
    llm_produced_answers,
    load_entity_map,
    summarize_checked_batch_records,
)
from oh_my_agent.common.qa_data import WebQSPQASample

RESULT_FILENAME = "checked_batch_eval.jsonl"
SUMMARY_FILENAME = "checked_batch_eval_summary.json"

DEFAULT_ENTITY_MAP = "data/resources/WebQSP/fbwq_full/mapped_entities.txt"

# 关键指标:回放 canonical 必须与录制 summary 吻合(浮点容差 1e-6)
VERIFY_KEYS = ("hit1", "hit_any", "macro_f1", "micro_f1", "exact_match", "citation_accuracy")


def _ladder_configs(score_margin: float) -> dict[str, dict[str, Any]]:
    """消融阶梯:逐层叠加后处理(check 恒为 canonical 的 constrained+hybrid)。"""
    return {
        "ablation_base": dict(drop_topic_self=False),
        "ablation_margin": dict(score_margin=score_margin, drop_topic_self=False),
        "ablation_margin_hop": dict(
            score_margin=score_margin, hop_filter=True, drop_topic_self=False
        ),
        "ablation_margin_hop_exp": dict(
            score_margin=score_margin, hop_filter=True,
            large_answer_expansion=True, drop_topic_self=False,
        ),
    }


def _sample_from_record(record: dict[str, Any]) -> WebQSPQASample:
    return WebQSPQASample(
        question_raw=record.get("question_raw", record.get("question", "")),
        question=record.get("question", ""),
        topic_mid=record.get("topic_mid", ""),
        gold_mids=list(record.get("gold_mids", [])),
    )


def _record_for_result(sample_index: int, sample: WebQSPQASample, result) -> dict[str, Any]:
    answer_metrics = compute_answer_metrics(
        result.pred_answer_disambiguated_mids, sample.gold_mids
    )
    faith = compute_faithfulness(
        cited_indices=set(result.final_accepted_path_indices)
        | set(result.relation_expanded_path_indices),
        golden_indices=label_golden_indices(result.raw_mmr_reason_paths, sample.gold_mids),
        pred_answers=llm_produced_answers(
            result.pred_answer_names,
            result.pred_answer_disambiguated_mids,
            result.large_answer_expanded_mids,
        ),
        path_entities=get_all_path_entities(result.named_mmr_reason_paths),
    )
    return build_eval_record(sample_index, sample, result, answer_metrics, faith)


def replay_config(
    records: list[dict[str, Any]],
    *,
    entity_map: dict[str, str],
    batch_size: int,
    expansion_min_answers: int,
    expansion_top_groups: int,
    hybrid_check: bool,
    **flags: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out_records: list[dict[str, Any]] = []
    for record in records:
        sample = _sample_from_record(record)
        result = replay_record(
            record,
            entity_map=entity_map,
            batch_size=batch_size,
            expansion_min_answers=expansion_min_answers,
            expansion_top_groups=expansion_top_groups,
            hybrid_check=hybrid_check,
            **flags,
        )
        out_records.append(
            _record_for_result(record.get("sample_index", 0), sample, result)
        )
    return out_records, summarize_checked_batch_records(out_records)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _verify_canonical(
    records, entity_map, batch_size, summary, *, score_margin,
    expansion_min_answers, expansion_top_groups, hybrid_check,
) -> None:
    _, replayed = replay_config(
        records,
        entity_map=entity_map,
        batch_size=batch_size,
        expansion_min_answers=expansion_min_answers,
        expansion_top_groups=expansion_top_groups,
        hybrid_check=hybrid_check,
        score_margin=score_margin,
        hop_filter=True,
        large_answer_expansion=True,
        drop_topic_self=True,
    )
    mismatches = []
    for key in VERIFY_KEYS:
        want, got = summary.get(key), replayed.get(key)
        if want is None or got is None:
            continue
        if abs(float(want) - float(got)) > 1e-6:
            mismatches.append(f"{key}: 录制={want} 回放={got}")
    if mismatches:
        raise SystemExit(
            "[ERROR] 回放 canonical 与录制 summary 不一致,拒绝写出消融档:\n  "
            + "\n  ".join(mismatches)
        )
    print("[OK] 回放校验通过:canonical 关键指标与录制逐项吻合")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="离线回放第五章后处理消融阶梯")
    parser.add_argument("--canonical_dir", required=True, help="canonical 评测输出子目录")
    parser.add_argument("--output_root", required=True, help="消融档写入的根目录")
    parser.add_argument("--entity_map", default=DEFAULT_ENTITY_MAP)
    parser.add_argument("--skip_verify", action="store_true")
    args = parser.parse_args(argv)

    canonical_dir = Path(args.canonical_dir)
    records_path = canonical_dir / RESULT_FILENAME
    summary_path = canonical_dir / SUMMARY_FILENAME
    if not records_path.exists() or not summary_path.exists():
        raise SystemExit(f"[ERROR] canonical 目录缺少 jsonl/summary: {canonical_dir}")

    records = _load_jsonl(records_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not records:
        raise SystemExit(f"[ERROR] canonical jsonl 为空: {records_path}")
    if "group_tails" not in records[0]:
        raise SystemExit(
            "[ERROR] canonical 录制缺少 group_tails 字段(旧版代码所跑),无法离线复现 "
            "expansion;请用当前代码重跑 canonical。"
        )

    batch_size = int(summary.get("batch_size", 20))
    score_margin = float(summary.get("score_margin") or 4.0)
    expansion_min_answers = int(summary.get("expansion_min_answers", 8))
    expansion_top_groups = int(summary.get("expansion_top_groups", 1))
    # canonical 的 check_mode 决定 check_tool_after_first 是否非 None,影响早停 → 回放必须一致
    hybrid_check = summary.get("check_mode", "hybrid-reject-list") == "hybrid-reject-list"

    print(f"[INFO] 录制样本数: {len(records)}  batch_size={batch_size} "
          f"score_margin={score_margin} expansion_top_groups={expansion_top_groups} "
          f"hybrid_check={hybrid_check}")

    if not args.skip_verify:
        _verify_canonical(
            records, summary=summary, entity_map=load_entity_map(args.entity_map),
            batch_size=batch_size, score_margin=score_margin,
            expansion_min_answers=expansion_min_answers,
            expansion_top_groups=expansion_top_groups, hybrid_check=hybrid_check,
        )

    entity_map = load_entity_map(args.entity_map)
    output_root = Path(args.output_root)
    for name, flags in _ladder_configs(score_margin).items():
        out_dir = output_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_records, out_summary = replay_config(
            records,
            entity_map=entity_map,
            batch_size=batch_size,
            expansion_min_answers=expansion_min_answers,
            expansion_top_groups=expansion_top_groups,
            hybrid_check=hybrid_check,
            **flags,
        )
        out_summary.update(
            {
                "replayed_from": str(records_path),
                "replay_config": name,
                "score_margin": flags.get("score_margin"),
                "hop_filter": flags.get("hop_filter", False),
                "large_answer_expansion": flags.get("large_answer_expansion", False),
                "topic_guard": flags.get("drop_topic_self", False),
                "batch_size": batch_size,
                "expansion_top_groups": expansion_top_groups,
            }
        )
        (out_dir / RESULT_FILENAME).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in out_records) + "\n",
            encoding="utf-8",
        )
        (out_dir / SUMMARY_FILENAME).write_text(
            json.dumps(out_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[WRITE] {name:24s} hit1={out_summary.get('hit1'):.4f} "
              f"macro_f1={out_summary.get('macro_f1'):.4f} "
              f"EM={out_summary.get('exact_match'):.4f} "
              f"cite={out_summary.get('citation_accuracy'):.4f}")

    print(f"[DONE] 消融阶梯写入 {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
