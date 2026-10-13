"""在无标签预测持久化后计算全指标与逐题配对收益。"""

import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np

from kgqa.core.answer_metrics import compute_answer_metrics, compute_faithfulness, label_golden_indices
from kgqa.pfit.eval import aggregate, is_rejection_text


class WorkflowScorer:
    @staticmethod
    def _align_mid_metrics(metrics: dict, answers: list, paths: list, gold_mids: list[str]) -> None:
        gold_mid_set = {str(m).lower().strip() for m in gold_mids if str(m).strip()}
        ans_norm_set = {str(a).lower().strip() for a in answers if str(a).strip()}
        pred_mids = set()
        for item in paths:
            c_tail = str(item.get("canonical_tail", "")).lower().strip()
            t_mid = str(item.get("tail_mid", "")).lower().strip()
            if c_tail in ans_norm_set or t_mid in ans_norm_set:
                pred_mids.add(t_mid)
        if not pred_mids:
            pred_mids = ans_norm_set
        mid_metrics = compute_answer_metrics(list(pred_mids), list(gold_mid_set))
        if mid_metrics["hit_any"] > metrics["hit_any"] or mid_metrics["f1"] > metrics["f1"]:
            for key in ("hit_any", "hit1", "exact_match", "f1", "precision", "recall"):
                metrics[key] = max(metrics[key], mid_metrics[key])

    @staticmethod
    def _metrics(answer: dict, paths: list, gold: list, gold_mids: list[str] | None = None) -> dict:
        canonical = deepcopy(paths)
        for item in canonical:
            if item.get("canonical_tail") and item.get("path"):
                item["path"][-1][2] = item["canonical_tail"]
        entities = {str(node) for item in canonical for h, _, t in item.get("path", []) for node in (h, t)}
        metrics = compute_answer_metrics(answer["answers"], gold)
        if gold_mids:
            WorkflowScorer._align_mid_metrics(metrics, answer["answers"], paths, gold_mids)
        metrics.update(compute_faithfulness(set(answer["cited_indices"]), label_golden_indices(canonical, gold),
                                           answer["answers"], entities))
        metrics["format_ok"] = answer["format_ok"]
        return metrics

    @staticmethod
    def _paired(before: list[dict], after: list[dict]) -> dict:
        rescue = sum(not b["hit_any"] and a["hit_any"] for b, a in zip(before, after, strict=True))
        harm = sum(b["hit_any"] and not a["hit_any"] for b, a in zip(before, after, strict=True))
        return {"rescue": rescue, "harm": harm, "net": rescue - harm}

    @classmethod
    def score_records(cls, records: list[dict], labels: dict[int, dict]) -> dict:
        before, after, groups, seen = [], [], defaultdict(list), set()
        for record in records:
            index = record["question"]["sample_index"]
            if index in seen or index not in labels or labels[index]["question"] != record["question"]["question"]:
                raise ValueError("预测重复、缺标签或问题身份不一致")
            seen.add(index)
            gold = labels[index]["golden"]
            gold_mids = labels[index].get("golden_mids")
            initial_paths = record.get("initial_paths", record["question"]["paths"])
            initial = cls._metrics(record["initial"], initial_paths, gold, gold_mids)
            final = cls._metrics(record["final"], record["final_paths"], gold, gold_mids)
            before.append(initial)
            after.append(final)
            groups[record["route"]["kind"]].append((initial, final, record))
        report = {"overall": aggregate(after), "baseline": aggregate(before),
                  "paired_hit": cls._paired(before, after), "by_route": {}}
        for kind, rows in groups.items():
            original, final, traces = zip(*rows)
            report["by_route"][kind] = {"baseline": aggregate(list(original)), "overall": aggregate(list(final)),
                                        "paired_hit": cls._paired(original, final),
                                        "trigger_n": sum(r["tool"] is not None for r in traces),
                                        "applied_n": sum(r["source"] != "baseline" for r in traces)}
        report["efficiency"] = cls._efficiency(records)
        report["source_counts"] = dict(Counter(r["source"] for r in records))
        report["reason_counts"] = dict(Counter(r.get("reason", "") for r in records))
        report["paired_hit_95ci"] = cls._bootstrap(before, after)
        report["scope"] = "同一全量拒答adapter实际初答与最终输出配对；已探索test上的实现验证"
        report["citation_definition"] = "沿用PFIT引用编号/gold尾匹配，符号答案无LLM引用，非完整语义证明"
        return report

    @staticmethod
    def _efficiency(records: list[dict]) -> dict:
        query_n = sum((r["tool"] or {}).get("stats", {}).get("query_n", 0) for r in records)
        network_n = sum((r["tool"] or {}).get("stats", {}).get("network_query_n", 0) for r in records)
        logical_calls = sum(r["pfit_calls"] for r in records)
        calls = []
        for row in records:
            if row.get("call_results"):
                calls.extend(row["call_results"])
            else:
                calls.append(row["initial"])
                if row["pfit_calls"] > 1:
                    calls.append(row["final"])
        prompt = sum(r["prompt_tokens"] for r in calls)
        completion = sum(r["completion_tokens"] for r in calls)
        latency = [r["elapsed_ms"] for r in records]
        return {"logical_pfit_calls": logical_calls, "prompt_tokens": prompt, "completion_tokens": completion,
                "kg_queries": query_n, "network_queries": network_n,
                "recorded_cache_hits": sum(c.get("cache_hit", False) for c in calls),
                "generation_cost_unknown_n": sum(bool(c.get("generation_error")) for c in calls),
                "sum_generation_elapsed_ms": sum(c["elapsed_ms"] for c in calls),
                "mean_workflow_wall_ms": float(np.mean(latency)) if latency else 0,
                "p95_workflow_wall_ms": float(np.percentile(latency, 95)) if latency else 0,
                "note": "前置初答来自本次实际生成缓存；wall时间为消费阶段，非冷启动端到端时延"}

    @staticmethod
    def _bootstrap(before: list[dict], after: list[dict]) -> list[float]:
        if not before:
            return [0.0, 0.0]
        delta = np.array([a["hit_any"] - b["hit_any"] for b, a in zip(before, after, strict=True)])
        generator = np.random.default_rng(17)
        draws = generator.integers(0, len(delta), size=(2000, len(delta)))
        return [float(x) for x in np.percentile(delta[draws].mean(axis=1), [2.5, 97.5])]

    @classmethod
    def score_file(cls, predictions: Path, labels_source: Path, allow_partial: bool = False) -> dict:
        labels = {r["sample_index"]: r for r in cls._read(labels_source)}
        records = cls._read(predictions)
        if not allow_partial and len(records) != len(labels):
            raise ValueError("正式评分要求完整样本；部分结果必须显式传 --allow_partial")
        report = cls.score_records(records, labels)
        report["partial"] = len(records) != len(labels)
        historical, fresh = [], []
        for row in records:
            source = labels[row["question"]["sample_index"]]
            answers = [name for name in source["llm_pred"] if not is_rejection_text(name)]
            historical.append(compute_answer_metrics(answers, source["golden"]))
            fresh.append(compute_answer_metrics(row["initial"]["answers"], source["golden"]))
        report["fresh_vs_historical_hit"] = cls._paired(historical, fresh)
        return report

    @staticmethod
    def _read(path: Path) -> list[dict]:
        with path.open() as handle:
            return [json.loads(line) for line in handle if line.strip()]
