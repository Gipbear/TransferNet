"""第三章 P2：配对区间与同口径检索效率证据。"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from experiments.common import ROOT
from kgqa.retrieve import engine
from kgqa.retrieve.datasets.registry import get_adapter
from kgqa.retrieve.shortest_path import ShortestPathParams, retrieve_shortest_paths_one
from utils.path_utils import path_answer_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="第三章 P2 配对区间与检索效率评测")
    parser.add_argument(
        "--config",
        default="experiments/configs/ch3/webqsp_transfernet_v1_p2.json",
        help="P2 预注册配置",
    )
    parser.add_argument("--project_dir", default=str(ROOT), help="项目根目录")
    parser.add_argument("--output_dir", required=True, help="机器可读证据输出目录")
    parser.add_argument(
        "--phase", choices=["statistics", "efficiency", "all"], default="all",
    )
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} 不是 JSON 对象")
                rows.append(row)
    return rows


def _index_rows(path: Path) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        if "sample_index" not in row:
            raise ValueError(f"{path} 存在缺少 sample_index 的记录")
        sample_index = int(row["sample_index"])
        if sample_index in indexed:
            raise ValueError(f"{path} 存在重复 sample_index={sample_index}")
        indexed[sample_index] = row
    return dict(sorted(indexed.items()))


def load_path_outcomes(path: Path) -> dict[int, dict[str, Any]]:
    outcomes: dict[int, dict[str, Any]] = {}
    for sample_index, row in _index_rows(path).items():
        gold = {str(value) for value in row.get("golden", [])}
        answer_sets = []
        for item in row.get("mmr_reason_paths", []):
            edges = item.get("path", [])
            if edges:
                nodes = [str(edges[0][0]), *(str(edge[2]) for edge in edges)]
                relations = [str(edge[1]) for edge in edges]
                answer_sets.append(path_answer_ids(nodes, relations))
        outcomes[sample_index] = {
            "question": str(row.get("question", "")),
            "answer_hit": float(any(answers & gold for answers in answer_sets)),
            "top1_hit": float(bool(answer_sets) and bool(answer_sets[0] & gold)),
        }
    return outcomes


def load_qa_outcomes(path: Path) -> dict[int, dict[str, Any]]:
    outcomes: dict[int, dict[str, Any]] = {}
    for sample_index, row in _index_rows(path).items():
        outcomes[sample_index] = {
            "question": str(row.get("question", "")),
            "hit1": float(row["hit1"]),
            "hit_any": float(row["hit_any"]),
            "macro_f1": float(row["f1"]),
        }
    return outcomes


def paired_metric_arrays(
    left: dict[int, dict[str, Any]],
    right: dict[int, dict[str, Any]],
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    if left.keys() != right.keys():
        missing_left = sorted(right.keys() - left.keys())[:5]
        missing_right = sorted(left.keys() - right.keys())[:5]
        raise ValueError(
            f"sample_index 不对齐: left 缺 {missing_left}, right 缺 {missing_right}"
        )
    left_values = []
    right_values = []
    for sample_index in left:
        if left[sample_index]["question"] != right[sample_index]["question"]:
            raise ValueError(f"sample_index={sample_index} 的 question 不一致")
        left_values.append(float(left[sample_index][metric]))
        right_values.append(float(right[sample_index][metric]))
    return np.asarray(left_values, dtype=np.float64), np.asarray(right_values, dtype=np.float64)


def paired_bootstrap_interval(
    left: np.ndarray,
    right: np.ndarray,
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    if left.ndim != 1 or right.ndim != 1 or len(left) != len(right) or not len(left):
        raise ValueError("配对 bootstrap 需要等长、非空的一维数组")
    if replicates <= 0:
        raise ValueError("bootstrap_replicates 必须为正整数")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level 必须位于 (0, 1)")

    differences = left - right
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(replicates, dtype=np.float64)
    batch_size = min(512, replicates)
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        indices = rng.integers(0, len(differences), size=(stop - start, len(differences)))
        bootstrap_means[start:stop] = differences[indices].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    ci_low, ci_high = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
    difference = float(differences.mean())
    return {
        "n": len(differences),
        "left_mean": float(left.mean()),
        "right_mean": float(right.mean()),
        "difference": difference,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "confidence_level": confidence_level,
        "replicates": replicates,
        "seed": seed,
        "ci_excludes_zero": bool(ci_low > 0.0 or ci_high < 0.0),
        "direction": "positive" if difference > 0 else "negative" if difference < 0 else "zero",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_statistics(config: dict[str, Any], project_dir: Path, output_dir: Path) -> dict[str, Any]:
    settings = config["statistics"]
    sources = {
        "path": {
            name: project_dir / relative for name, relative in settings["path_inputs"].items()
        },
        "qa": {
            name: project_dir / relative for name, relative in settings["qa_inputs"].items()
        },
    }
    loaded = {
        "path": {name: load_path_outcomes(path) for name, path in sources["path"].items()},
        "qa": {name: load_qa_outcomes(path) for name, path in sources["qa"].items()},
    }
    comparisons = []
    interval_index = 0
    for comparison in settings["comparisons"]:
        family = comparison["family"]
        left_name = comparison["left"]
        right_name = comparison["right"]
        intervals = {}
        for metric in comparison["metrics"]:
            left, right = paired_metric_arrays(
                loaded[family][left_name], loaded[family][right_name], metric,
            )
            intervals[metric] = paired_bootstrap_interval(
                left,
                right,
                replicates=int(settings["bootstrap_replicates"]),
                confidence_level=float(settings["confidence_level"]),
                seed=int(settings["seed"]) + interval_index,
            )
            interval_index += 1
        comparisons.append({**comparison, "intervals": intervals})

    result = {
        "schema_version": 1,
        "kind": "ch3_paired_bootstrap",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "estimand": config["estimand"],
        "method": "paired percentile bootstrap",
        "decision_rule": "CI excludes zero; no post-hoc p-value is reported",
        "comparisons": comparisons,
        "pending_comparisons": settings.get("pending_comparisons", []),
        "sources": {
            family: {name: _fingerprint(path) for name, path in paths.items()}
            for family, paths in sources.items()
        },
    }
    _write_json(output_dir / "paired_bootstrap.json", result)
    return result


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _git_state(project_dir: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_dir, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=project_dir, check=True,
        capture_output=True, text=True,
    ).stdout.strip())
    return {"commit": commit, "dirty": dirty}


def _environment(project_dir: Path) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "pytorch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "git": _git_state(project_dir),
    }


def _method_runner(
    method: dict[str, Any], adapter, bundle,
) -> Callable[[Any, engine.RetrievalDiagnostics | None], Any]:
    if method["kind"] == "shortest_path":
        params = ShortestPathParams(**method["params"])

        def run_shortest(sample, diagnostics=None):
            return retrieve_shortest_paths_one(
                sample,
                adapter.kg_edge_source(sample),
                bundle.meta.id2ent,
                bundle.meta.id2rel,
                params=params,
                diagnostics=diagnostics,
            )

        return run_shortest
    if method["kind"] == "score_beam":
        params = dict(method["params"])

        def run_score_beam(sample, diagnostics=None):
            return engine.retrieve_one(
                sample,
                adapter.kg_edge_source(sample),
                bundle.meta.id2ent,
                bundle.meta.id2rel,
                diagnostics=diagnostics,
                **params,
            )

        return run_score_beam
    raise ValueError(f"未知效率方法 kind: {method['kind']}")


def _measure_memory(samples: list, run_one: Callable) -> float:
    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()
    for sample in samples:
        result = run_one(sample, None)
        del result
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


def run_efficiency(config: dict[str, Any], project_dir: Path, output_dir: Path) -> dict[str, Any]:
    settings = config["efficiency"]
    cache_path = project_dir / settings["cache"]
    input_dir = project_dir / settings["input_dir"]
    adapter = get_adapter(config["dataset"], input_dir=str(input_dir))
    bundle = adapter.score_loader().load(str(cache_path))
    limit = int(settings.get("sample_limit", 0))
    samples = bundle.samples[:limit] if limit else bundle.samples
    if not samples:
        raise ValueError("效率评测没有可用样本")

    adapter.kg_edge_source(samples[0])
    warmup_n = min(int(settings["warmup_samples"]), len(samples))
    repeats = int(settings["timing_repeats"])
    partial_path = output_dir / "efficiency_partial.json"
    completed: dict[str, dict[str, Any]] = {}
    if partial_path.exists():
        completed = {
            item["id"]: item
            for item in json.loads(partial_path.read_text(encoding="utf-8")).get("methods", [])
        }
    method_results = []
    for method in settings["methods"]:
        if method["id"] in completed:
            method_results.append(completed[method["id"]])
            print(f"[P2] {method['id']} 复用已完成的中间结果", flush=True)
            continue
        run_one = _method_runner(method, adapter, bundle)
        for sample in samples[:warmup_n]:
            run_one(sample, None)

        latencies_ms: list[float] = []
        repeat_seconds: list[float] = []
        expanded_states: list[int] = []
        candidate_paths: list[int] = []
        final_paths: list[int] = []
        for repeat_index in range(repeats):
            repeat_started = time.perf_counter()
            for sample in samples:
                diagnostics = engine.RetrievalDiagnostics()
                started = time.perf_counter_ns()
                result = run_one(sample, diagnostics)
                latencies_ms.append((time.perf_counter_ns() - started) / 1_000_000)
                if repeat_index == 0:
                    expanded_states.append(diagnostics.expanded_states)
                    candidate_paths.append(diagnostics.candidate_paths)
                    final_paths.append(diagnostics.final_paths)
                del result
            repeat_seconds.append(time.perf_counter() - repeat_started)
            print(
                f"[P2] {method['id']} timing {repeat_index + 1}/{repeats}: "
                f"{repeat_seconds[-1]:.2f}s",
                flush=True,
            )
        memory_samples = samples[:warmup_n]
        peak_memory_mb = _measure_memory(memory_samples, run_one)
        method_results.append({
            "id": method["id"],
            "kind": method["kind"],
            "params": method["params"],
            "samples": len(samples),
            "timing_measurements": len(latencies_ms),
            "timing_repeats": repeats,
            "mean_ms": statistics.fmean(latencies_ms),
            "p50_ms": _percentile(latencies_ms, 50),
            "p95_ms": _percentile(latencies_ms, 95),
            "repeat_seconds": repeat_seconds,
            "python_incremental_peak_mb": peak_memory_mb,
            "memory_samples": len(memory_samples),
            "mean_expanded_states": statistics.fmean(expanded_states),
            "mean_candidate_paths": statistics.fmean(candidate_paths),
            "mean_final_paths": statistics.fmean(final_paths),
        })
        _write_json(partial_path, {"methods": method_results})
        print(f"[P2] {method['id']} memory pass: {peak_memory_mb:.2f} MiB", flush=True)

    result = {
        "schema_version": 1,
        "kind": "ch3_retrieval_efficiency",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "environment": _environment(project_dir),
        "measurement_scope": {
            "start": "immediately before one-sample retrieval",
            "end": "after RetrieveResult construction and path serialization",
            "includes_score_cache_load": False,
            "includes_kg_initialization": False,
            "includes_llm": False,
            "sample_order": "score-cache order, identical for every method and repeat",
            "warmup_samples_per_method": warmup_n,
            "timing_repeats": repeats,
            "latency_aggregation": "all per-sample observations across repeats",
            "memory_metric": "tracemalloc incremental Python allocation peak in a separate 100-sample pass",
            "expanded_states_definition": (
                "score methods: beam states whose outgoing adjacency is scanned; "
                "SP: conceptual BFS states through the available hop budget"
            ),
            "candidate_paths_definition": "unique/pre-selection paths before the final path budget",
            "final_paths_definition": "serialized paths after selection and loopback removal",
        },
        "cache": _fingerprint(cache_path),
        "methods": method_results,
    }
    _write_json(output_dir / "efficiency.json", result)
    partial_path.unlink(missing_ok=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_dir = Path(args.project_dir).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_dir / config_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_dir / output_dir
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "p2_config_snapshot.json", config)
    if args.phase in {"statistics", "all"}:
        run_statistics(config, project_dir, output_dir)
    if args.phase in {"efficiency", "all"}:
        run_efficiency(config, project_dir, output_dir)
    print(f"[P2] 证据已写入 {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
