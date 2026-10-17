"""流式核对第五章既有指标并统计同一候选池的路径尾实体覆盖。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from kgqa.core.answer_metrics import aggregate_metrics, compute_answer_metrics, norm_entity
from kgqa.core.qa_formats import clean_question_text, load_webqsp_qa_samples


class EvidencePath(BaseModel):
    model_config = ConfigDict(frozen=True)
    path: list[tuple[str, str, str]]


class AuditRecord(BaseModel):
    model_config = ConfigDict(frozen=True)
    sample_index: int
    question: str
    gold_mids: list[str]
    pred_answer_disambiguated_mids: list[str]
    raw_mmr_reason_paths: list[EvidencePath]
    citation_accuracy: float
    citation_recall: float
    hallucination_rate: float
    format_ok: bool


class BenchmarkAudit:
    def __init__(self, source: Path, qa: Path):
        self.source = source
        self.samples = load_webqsp_qa_samples(str(qa))
        self.qa_hash = hashlib.sha256(qa.read_bytes()).hexdigest()

    def run(self, output: Path) -> None:
        seen: set[int] = set()
        metrics = []
        coverage = {20: {"hit_n": 0, "covered_gold_n": 0}, 50: {"hit_n": 0, "covered_gold_n": 0}}
        rescued: list[int] = []
        total_gold = 0
        digest = hashlib.sha256()
        with (self.source / "checked_batch_eval.jsonl").open("rb") as handle:
            for line in handle:
                digest.update(line)
                record = AuditRecord.model_validate_json(line)
                self._verify_sample(record, seen)
                gold = {norm_entity(mid) for mid in record.gold_mids}
                total_gold += len(gold)
                hits = []
                for budget, counts in coverage.items():
                    tails = {norm_entity(p.path[-1][2]) for p in record.raw_mmr_reason_paths[:budget] if p.path}
                    found = gold & tails
                    counts["hit_n"] += int(bool(found))
                    counts["covered_gold_n"] += len(found)
                    hits.append(bool(found))
                if hits == [False, True]:
                    rescued.append(record.sample_index)
                row = compute_answer_metrics(record.pred_answer_disambiguated_mids, record.gold_mids)
                row.update(record.model_dump(include={
                    "citation_accuracy", "citation_recall", "hallucination_rate", "format_ok",
                }))
                metrics.append(row)
        if seen != set(range(len(self.samples))):
            raise ValueError("原始轨迹没有完整覆盖当前QA样本索引")
        summary = aggregate_metrics(metrics)
        expected = json.loads((self.source / "checked_batch_eval_summary.json").read_text(encoding="utf-8"))
        mismatches = {key: [expected.get(key), value] for key, value in summary.items() if expected.get(key) != value}
        if mismatches:
            raise ValueError(f"指标重算与原summary不一致: {mismatches}")
        result = {
            "source": str(self.source.resolve()), "qa_sha256": self.qa_hash,
            "trace_sha256": digest.hexdigest(), "metric_reproduction": "passed", "summary": summary,
            "coverage_scope": "同一beam50池的路径尾实体覆盖，不等同于独立beam20检索",
            "coverage": coverage, "gold_entities_total": total_gold, "rescued_sample_indices": rescued,
            "pending": ["独立beam20一致性", "逐样本agent回放", "新模型推理与补全复核"],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

    def _verify_sample(self, record: AuditRecord, seen: set[int]) -> None:
        index = record.sample_index
        if index in seen or not 0 <= index < len(self.samples):
            raise ValueError(f"无效或重复sample_index: {index}")
        sample = self.samples[index]
        if set(record.gold_mids) != set(sample.gold_mids):
            raise ValueError(f"gold与当前QA不一致: {index}")
        if clean_question_text(record.question) != sample.question:
            raise ValueError(f"问题与当前QA不一致: {index}")
        seen.add(index)


def main() -> None:
    parser = argparse.ArgumentParser(description="第五章CPU指标复算与路径覆盖核对")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--qa", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    BenchmarkAudit(args.source, args.qa).run(args.output)


if __name__ == "__main__":
    main()
