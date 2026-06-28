"""Collect stop-policy sweep summaries into a compact comparison report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

SUMMARY_CSV = "stop_policy_sweep_summary.csv"


def _read_rows(sweep_dir: Path) -> list[dict[str, Any]]:
    path = sweep_dir / SUMMARY_CSV
    if not path.exists():
        raise FileNotFoundError(f"missing {SUMMARY_CSV}: {sweep_dir}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    return float(value) if value not in {None, ""} else 0.0


def _int(row: dict[str, Any], key: str) -> int:
    value = row.get(key)
    return int(float(value)) if value not in {None, ""} else 0


def _complete(row: dict[str, Any]) -> bool:
    return str(row.get("complete_support", "")).lower() == "true"


def _combined_rows(sweep_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sweep_dir in sweep_dirs:
        label = sweep_dir.name
        for row in _read_rows(sweep_dir):
            row = dict(row)
            row["sweep"] = label
            row["sweep_dir"] = str(sweep_dir)
            rows.append(row)
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    preferred = [
        "sweep",
        "policy",
        "complete_support",
        "n",
        "unsupported_n",
        "macro_f1",
        "exact_match",
        "hit1",
        "hit_any",
        "citation_accuracy",
        "avg_batches_used",
        "avg_checked_paths",
        "avg_final_answer_count",
        "mixed_stop_ratio",
        "max_batches",
        "no_all_wrong_after_answer_stop",
        "stop_after_no_new_batches",
        "stop_reason_counts",
        "sweep_dir",
    ]
    fieldnames = preferred + sorted({key for row in rows for key in row} - set(preferred))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_metric(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if value in {None, ""}:
        return ""
    return f"{float(value):.4f}"


def _markdown(rows: list[dict[str, Any]], *, top_k: int) -> str:
    lines = [
        "# Stop-policy sweep comparison",
        "",
        "Only `complete_support=true` rows are ranked as directly comparable. Rows with unsupported samples require a fuller source trace before drawing conclusions.",
        "",
        "## Best complete policy per sweep",
        "",
        "| sweep | policy | macro_f1 | EM | hit1 | avg_batches | avg_answers | unsupported |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for sweep in sorted({row["sweep"] for row in rows}):
        complete = [row for row in rows if row["sweep"] == sweep and _complete(row)]
        if not complete:
            lines.append(f"| {sweep} | N/A |  |  |  |  |  |  |")
            continue
        best = max(complete, key=lambda row: _float(row, "macro_f1"))
        lines.append(
            "| {sweep} | `{policy}` | {macro_f1} | {em} | {hit1} | {batches} | {answers} | {unsupported} |".format(
                sweep=sweep,
                policy=best.get("policy", ""),
                macro_f1=_format_metric(best, "macro_f1"),
                em=_format_metric(best, "exact_match"),
                hit1=_format_metric(best, "hit1"),
                batches=_format_metric(best, "avg_batches_used"),
                answers=_format_metric(best, "avg_final_answer_count"),
                unsupported=best.get("unsupported_n", ""),
            )
        )

    ranked = sorted(
        [row for row in rows if _complete(row)],
        key=lambda row: (_float(row, "macro_f1"), _float(row, "exact_match")),
        reverse=True,
    )[:top_k]
    lines.extend(
        [
            "",
            f"## Top {len(ranked)} complete policies overall",
            "",
            "| sweep | policy | macro_f1 | EM | hit1 | avg_batches | avg_answers | stop_reason_counts |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in ranked:
        lines.append(
            "| {sweep} | `{policy}` | {macro_f1} | {em} | {hit1} | {batches} | {answers} | `{stops}` |".format(
                sweep=row.get("sweep", ""),
                policy=row.get("policy", ""),
                macro_f1=_format_metric(row, "macro_f1"),
                em=_format_metric(row, "exact_match"),
                hit1=_format_metric(row, "hit1"),
                batches=_format_metric(row, "avg_batches_used"),
                answers=_format_metric(row, "avg_final_answer_count"),
                stops=row.get("stop_reason_counts", ""),
            )
        )

    unsupported_count = sum(1 for row in rows if _int(row, "unsupported_n") > 0)
    lines.extend(
        [
            "",
            "## Support coverage",
            "",
            f"- Total policy rows: {len(rows)}",
            f"- Complete rows: {sum(1 for row in rows if _complete(row))}",
            f"- Rows with unsupported samples: {unsupported_count}",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_dirs", nargs="+", help="Directories containing stop_policy_sweep_summary.csv")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_k", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sweep_dirs = [Path(item) for item in args.sweep_dirs]
    rows = _combined_rows(sweep_dirs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output_dir / "stop_policy_compare.csv")
    (output_dir / "stop_policy_compare.md").write_text(
        _markdown(rows, top_k=args.top_k), encoding="utf-8"
    )
    print(f"[WRITE] {output_dir / 'stop_policy_compare.csv'}")
    print(f"[WRITE] {output_dir / 'stop_policy_compare.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
