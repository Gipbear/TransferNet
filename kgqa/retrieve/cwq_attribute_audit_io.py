"""写入 CWQ 属性证据审计产物。"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from kgqa.retrieve.cwq_attribute_audit import AuditConfig, LocalSampleAudit, RemoteSampleAudit, StrictSampleAudit


def write_jsonl(path: Path, records: Sequence[LocalSampleAudit | RemoteSampleAudit | StrictSampleAudit]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def write_summary(
    config: AuditConfig,
    local: tuple[LocalSampleAudit, ...],
    remote: tuple[RemoteSampleAudit, ...],
    strict: tuple[StrictSampleAudit, ...],
) -> None:
    operator_hits = Counter(record.operator for record in remote if record.evaluation.oracle_hit)
    type_hits = Counter(record.compositionality_type for record in remote if record.evaluation.oracle_hit)
    strict_operator_hits = Counter(record.operator for record in strict if record.oracle_hit)
    strict_type_hits = Counter(record.compositionality_type for record in strict if record.oracle_hit)
    union_pairs = [
        (remote_record, strict_record)
        for remote_record, strict_record in zip(remote, strict, strict=True)
        if remote_record.evaluation.oracle_hit or strict_record.oracle_hit
    ]
    union_operator_hits = Counter(remote_record.operator for remote_record, _ in union_pairs)
    union_type_hits = Counter(remote_record.compositionality_type for remote_record, _ in union_pairs)
    summary = {
        "sample_count": len(remote),
        "local_relation_vocab_complete_sample_count": sum(record.relation_vocab_complete for record in local),
        "local_any_first_hop_path_sample_count": sum(record.first_hop_candidate_count > 0 for record in local),
        "local_any_complete_value_sample_count": sum(record.complete_value_candidate_count > 0 for record in local),
        "remote_gold_candidate_present_sample_count": sum(record.evaluation.gold_present for record in remote),
        "remote_any_comparable_value_sample_count": sum(
            record.evaluation.comparable_candidate_count > 0 for record in remote
        ),
        "remote_gold_has_value_sample_count": sum(record.evaluation.gold_has_value for record in remote),
        "remote_oracle_rule_hit_sample_count": sum(record.evaluation.oracle_hit for record in remote),
        "remote_oracle_rule_hit_by_operator": dict(sorted(operator_hits.items())),
        "remote_oracle_rule_hit_by_type": dict(sorted(type_hits.items())),
        "strict_oracle_nonempty_sample_count": sum(bool(record.selected_mids) for record in strict),
        "strict_oracle_hit_sample_count": sum(record.oracle_hit for record in strict),
        "strict_oracle_hit_by_operator": dict(sorted(strict_operator_hits.items())),
        "strict_oracle_hit_by_type": dict(sorted(strict_type_hits.items())),
        "union_oracle_hit_sample_count": len(union_pairs),
        "union_oracle_hit_by_operator": dict(sorted(union_operator_hits.items())),
        "union_oracle_hit_by_type": dict(sorted(union_type_hits.items())),
        "gold_candidate_present_but_unrecovered_sample_count": sum(
            remote_record.evaluation.gold_present
            and not remote_record.evaluation.oracle_hit
            and not strict_record.oracle_hit
            for remote_record, strict_record in zip(remote, strict, strict=True)
        ),
    }
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
