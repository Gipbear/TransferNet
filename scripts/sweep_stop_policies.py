"""Offline sweep of checked-batch stop policies from recorded agent traces.

The script replays an existing ``checked_batch_eval.jsonl`` through the real
``CheckedBatchWebQAgent`` with mock retrieval/answer/check tools. It never calls
LLMs or path servers. Policies that need batches beyond the recorded trace are
marked unsupported; collect a full trace with ``--no_early_stop`` and
``--no_all_wrong_after_answer_stop`` when you need to evaluate less aggressive
continuation policies.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oh_my_agent.agent.checked_batch_replay import _ReplaySession
from oh_my_agent.common import (
    build_eval_record,
    cited_indices_for_answers,
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


def _load_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
        if limit and len(records) >= limit:
            break
    return records


def _sample_from_record(record: dict[str, Any]) -> WebQSPQASample:
    return WebQSPQASample(
        question_raw=record.get("question_raw", record.get("question", "")),
        question=record.get("question", ""),
        topic_mid=record.get("topic_mid", ""),
        gold_mids=list(record.get("gold_mids", [])),
    )


def _record_for_result(sample_index: int, sample: WebQSPQASample, result) -> dict[str, Any]:
    answer_metrics = compute_answer_metrics(
        result.pred_answer_disambiguated_mids,
        sample.gold_mids,
    )
    faith = compute_faithfulness(
        cited_indices=cited_indices_for_answers(
            set(result.final_accepted_path_indices)
            | set(result.relation_expanded_path_indices),
            result.raw_mmr_reason_paths,
            result.pred_answer_disambiguated_mids,
        ),
        golden_indices=label_golden_indices(result.raw_mmr_reason_paths, sample.gold_mids),
        pred_answers=llm_produced_answers(
            result.pred_answer_names,
            result.pred_answer_disambiguated_mids,
            result.large_answer_expanded_mids,
        ),
        path_entities=get_all_path_entities(result.named_mmr_reason_paths),
    )
    return build_eval_record(sample_index, sample, result, answer_metrics, faith)


def _parse_ratio_list(raw: str) -> list[float | None]:
    values: list[float | None] = []
    for item in raw.replace(",", " ").split():
        key = item.strip().lower()
        if key in {"off", "none", "null", "no", "disabled"}:
            values.append(None)
        elif "/" in key:
            numerator, denominator = key.split("/", 1)
            values.append(float(numerator) / float(denominator))
        else:
            values.append(float(key))
    return values


def _parse_int_or_none_list(raw: str, *, none_words: set[str]) -> list[int | None]:
    values: list[int | None] = []
    for item in raw.replace(",", " ").split():
        key = item.strip().lower()
        if key in none_words:
            values.append(None)
        else:
            value = int(key)
            if value <= 0:
                raise ValueError(f"positive integer expected, got {item!r}")
            values.append(value)
    return values


def _parse_all_wrong_modes(raw: str) -> list[bool]:
    """Return no_all_wrong_after_answer_stop values."""
    values: list[bool] = []
    for item in raw.replace(",", " ").split():
        key = item.strip().lower()
        if key in {"on", "stop", "enabled", "true"}:
            values.append(False)
        elif key in {"off", "nostop", "disabled", "false"}:
            values.append(True)
        else:
            raise ValueError(f"unknown all-wrong mode: {item!r}")
    return values


def _mode_bool(raw: str, summary: dict[str, Any], key: str, default: bool) -> bool:
    if raw == "on":
        return True
    if raw == "off":
        return False
    value = summary.get(key)
    return default if value is None else bool(value)


def _score_margin(raw: str, summary: dict[str, Any]) -> float | None:
    if raw == "auto":
        value = summary.get("score_margin")
        return None if value is None else float(value)
    if raw.lower() in {"none", "off", "null"}:
        return None
    return float(raw)


def _base_run_flags(args: argparse.Namespace, summary: dict[str, Any]) -> dict[str, Any]:
    flags: dict[str, Any] = {
        "score_margin": _score_margin(args.score_margin, summary),
        "hop_filter": _mode_bool(args.hop_filter, summary, "hop_filter", False),
        "large_answer_expansion": _mode_bool(
            args.large_answer_expansion, summary, "large_answer_expansion", False
        ),
        # eval_checked_batch_agent defaults to topic guard on, and older summaries
        # do not carry an explicit key for it.
        "drop_topic_self": _mode_bool(args.topic_guard, summary, "topic_guard", True),
        "expansion_min_answers": int(summary.get("expansion_min_answers") or 8),
        "expansion_top_groups": int(summary.get("expansion_top_groups") or 1),
    }
    if summary.get("relation_expansion") is not None:
        flags["enable_relation_expansion"] = bool(summary.get("relation_expansion"))
    return flags


def _policy_name(
    *,
    mixed_stop_ratio: float | None,
    max_batches: int | None,
    no_all_wrong_after_answer_stop: bool,
    stop_after_no_new_batches: int | None,
) -> str:
    if mixed_stop_ratio is None:
        mixed = "mixoff"
    else:
        mixed = f"mix{mixed_stop_ratio:g}".replace(".", "p")
    max_part = f"max{max_batches}" if max_batches is not None else "maxall"
    all_wrong = "awoff" if no_all_wrong_after_answer_stop else "awon"
    no_new = (
        f"nonew{stop_after_no_new_batches}"
        if stop_after_no_new_batches is not None
        else "nonewoff"
    )
    return "_".join([mixed, max_part, all_wrong, no_new])


def _build_policies(args: argparse.Namespace) -> list[dict[str, Any]]:
    ratios = _parse_ratio_list(args.mixed_stop_ratios)
    max_batches_values = _parse_int_or_none_list(
        args.max_batches, none_words={"all", "none", "null", "off"}
    )
    all_wrong_modes = _parse_all_wrong_modes(args.all_wrong_modes)
    no_new_values = _parse_int_or_none_list(
        args.no_new_batches, none_words={"none", "null", "off", "disabled"}
    )

    policies: dict[str, dict[str, Any]] = {}
    for ratio, max_batches, no_all_wrong, no_new in itertools.product(
        ratios, max_batches_values, all_wrong_modes, no_new_values
    ):
        name = _policy_name(
            mixed_stop_ratio=ratio,
            max_batches=max_batches,
            no_all_wrong_after_answer_stop=no_all_wrong,
            stop_after_no_new_batches=no_new,
        )
        policies[name] = {
            "policy": name,
            "mixed_stop_ratio": ratio,
            "max_batches": max_batches,
            "stop_after_no_new_batches": no_new,
            "no_all_wrong_after_answer_stop": no_all_wrong,
            "no_early_stop": ratio is None,
        }
    return list(policies.values())


def _replay_policy(
    records: list[dict[str, Any]],
    *,
    session: _ReplaySession,
    base_flags: dict[str, Any],
    policy: dict[str, Any],
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out_records: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    run_flags = dict(base_flags)
    run_flags.update(
        {
            "batch_size": batch_size,
            "mixed_stop_ratio": policy["mixed_stop_ratio"],
            "max_batches": policy["max_batches"],
            "stop_after_no_new_batches": policy["stop_after_no_new_batches"],
            "no_all_wrong_after_answer_stop": policy[
                "no_all_wrong_after_answer_stop"
            ],
            "no_early_stop": policy["no_early_stop"],
        }
    )
    for record in records:
        sample = _sample_from_record(record)
        try:
            result = session.replay(record, allow_prefix=True, **run_flags)
        except (IndexError, ValueError) as exc:
            unsupported.append(
                {
                    "sample_index": record.get("sample_index"),
                    "error": str(exc),
                    "recorded_batches": len(record.get("iterations", [])),
                }
            )
            continue
        out_records.append(
            _record_for_result(record.get("sample_index", 0), sample, result)
        )
    return out_records, unsupported


def _write_policy_outputs(
    *,
    output_dir: Path,
    policy: dict[str, Any],
    summary: dict[str, Any],
    records: list[dict[str, Any]],
    unsupported: list[dict[str, Any]],
    write_records: bool,
) -> None:
    policy_dir = output_dir / policy["policy"]
    policy_dir.mkdir(parents=True, exist_ok=True)
    (policy_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if unsupported:
        (policy_dir / "unsupported_samples.jsonl").write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in unsupported) + "\n",
            encoding="utf-8",
        )
    if write_records:
        (policy_dir / RESULT_FILENAME).write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
            encoding="utf-8",
        )


def _summary_row(summary: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "policy",
        "complete_support",
        "source_n",
        "n",
        "unsupported_n",
        "hit1",
        "hit_any",
        "macro_f1",
        "micro_f1",
        "exact_match",
        "citation_accuracy",
        "avg_batches_used",
        "avg_checked_paths",
        "avg_accepted_paths",
        "avg_final_answer_count",
        "mixed_stop_ratio",
        "max_batches",
        "stop_after_no_new_batches",
        "no_all_wrong_after_answer_stop",
        "stop_reason_counts",
    ]
    row = {key: summary.get(key) for key in keys}
    row["stop_reason_counts"] = json.dumps(
        row.get("stop_reason_counts") or {}, ensure_ascii=False, sort_keys=True
    )
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--entity_map", default=DEFAULT_ENTITY_MAP)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--write_records", action="store_true")
    parser.add_argument("--require_complete_support", action="store_true")
    parser.add_argument("--mixed_stop_ratios", default="0,0.1,1/3,0.5,off")
    parser.add_argument("--max_batches", default="1,2,3,all")
    parser.add_argument("--all_wrong_modes", default="on,off")
    parser.add_argument("--no_new_batches", default="none")
    parser.add_argument("--score_margin", default="auto")
    parser.add_argument("--hop_filter", choices=["auto", "on", "off"], default="auto")
    parser.add_argument(
        "--large_answer_expansion", choices=["auto", "on", "off"], default="auto"
    )
    parser.add_argument("--topic_guard", choices=["auto", "on", "off"], default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_dir = Path(args.source_dir)
    records_path = source_dir / RESULT_FILENAME
    summary_path = source_dir / SUMMARY_FILENAME
    if not records_path.exists() or not summary_path.exists():
        raise SystemExit(f"source_dir lacks {RESULT_FILENAME}/{SUMMARY_FILENAME}: {source_dir}")

    records = _load_jsonl(records_path, limit=args.limit)
    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not records:
        raise SystemExit(f"no records loaded from {records_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = int(source_summary.get("batch_size") or records[0]["iterations"][0]["batch_size"])
    hybrid_check = source_summary.get("check_mode") == "hybrid-reject-list"
    base_flags = _base_run_flags(args, source_summary)
    if base_flags["large_answer_expansion"] and any("group_tails" not in record for record in records):
        raise SystemExit(
            "source records lack group_tails, so large_answer_expansion cannot be "
            "replayed faithfully. Rerun the source trace with current code, or pass "
            "--large_answer_expansion off for a no-expansion stop-policy sweep."
        )
    policies = _build_policies(args)

    print(
        f"[INFO] source={source_dir} records={len(records)} batch_size={batch_size} "
        f"hybrid_check={hybrid_check} policies={len(policies)}"
    )
    session = _ReplaySession(load_entity_map(args.entity_map), hybrid_check=hybrid_check)

    summaries: list[dict[str, Any]] = []
    for policy in policies:
        out_records, unsupported = _replay_policy(
            records,
            session=session,
            base_flags=base_flags,
            policy=policy,
            batch_size=batch_size,
        )
        if args.require_complete_support and unsupported:
            raise SystemExit(
                f"policy {policy['policy']} unsupported for {len(unsupported)} samples; "
                "use a fuller source trace or drop --require_complete_support"
            )
        summary = summarize_checked_batch_records(out_records)
        summary.update(policy)
        summary.update(
            {
                "source_dir": str(source_dir),
                "source_records_path": str(records_path),
                "source_n": len(records),
                "unsupported_n": len(unsupported),
                "complete_support": len(unsupported) == 0,
                "batch_size": batch_size,
                "base_flags": base_flags,
            }
        )
        summaries.append(summary)
        _write_policy_outputs(
            output_dir=output_dir,
            policy=policy,
            summary=summary,
            records=out_records,
            unsupported=unsupported,
            write_records=args.write_records,
        )
        metric = summary.get("macro_f1")
        metric_text = "NA" if metric is None else f"{metric:.4f}"
        print(
            f"[WRITE] {policy['policy']:34s} n={summary.get('n', 0):4} "
            f"unsupported={len(unsupported):4} macro_f1={metric_text}"
        )

    rows = [_summary_row(summary) for summary in summaries]
    csv_path = output_dir / "stop_policy_sweep_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "stop_policy_sweep_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[DONE] wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
