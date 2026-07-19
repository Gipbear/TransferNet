"""统一评测 CLI：检索 + 骨干模型/路径终点指标。"""
from __future__ import annotations

import argparse
import json
import os
import time

from tqdm import tqdm

from kgqa.retrieve.cli.retrieve import (
    build_backend,
    build_parser as _retrieve_parser,
    run_retrieval,
)
from kgqa.retrieve.eval.answer_eval import answer_record, answer_summary
from kgqa.retrieve.eval.path_eval import path_summary
from kgqa.retrieve.engine import validate_score_scheme
from kgqa.runtime import configure_runtime, emit_event, update_progress


def build_parser() -> argparse.ArgumentParser:
    p = _retrieve_parser()
    p.description = "kgqa 统一评测"
    p.add_argument("--summary", default=None, help="summary.json 输出路径")
    p.add_argument("--jobs_file", default=None,
                   help="批量评测任务 JSON；复用同一离线缓存与数据集适配器")
    return p


def _gold_strings(sample, adapter, id2ent, gold_key: str) -> set[str]:
    """按 spec.gold_key 统一 gold 口径。

    - mid: gold_ids 是整数实体 id → 经 id2ent 映射成 MID，与 prediction（MID 键）/
      路径尾（id2ent[tail]=MID）同口径。
    - name: 整数 gold_ids 同样先经 id2ent 还原（MetaQA id2ent 即实体名），再过
      adapter.entity_name，与 prediction（名称键）同口径。
    """
    out: set[str] = set()
    for g in sample.gold_ids:
        base = id2ent.get(int(g), str(g)) if isinstance(g, int) else str(g)
        out.add(adapter.entity_name(base) if gold_key == "name" else base)
    return out


def write_results(results, output: str | None) -> None:
    if not output:
        return
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        for result in results:
            fh.write(json.dumps({
                "sample_index": result.sample_index, "question": result.question,
                "topics": result.topics, "hop": result.hop, "golden": result.golden,
                "mmr_reason_paths": result.paths, "prediction": result.prediction,
            }, ensure_ascii=False) + "\n")


def evaluate_results(
    backend, results, summary_path: str | None, retrieve_params: dict | None = None,
) -> dict:
    adapter = backend.adapter
    spec = adapter.metric_spec()
    id2ent = backend.bundle.meta.id2ent
    id2rel = backend.bundle.meta.id2rel

    gold_by_index: dict[int, set[str]] = {}
    backbone_records = []
    for r, sample in zip(results, backend.bundle.samples):
        gold = _gold_strings(sample, adapter, id2ent, spec.gold_key)
        gold_by_index[r.sample_index] = gold
        # prediction 直接来自骨干模型的实体分数，不受路径重建和重排序影响。
        backbone_prediction = list(r.prediction.keys())
        backbone_records.append(answer_record(
            pred=backbone_prediction, gold=sorted(gold), hop=sample.hop, format_ok=True,
        ))

    summary = {
        "backbone": answer_summary(backbone_records, spec),
        "path": path_summary(results, gold_by_index, spec, id2rel=id2rel),
        "n": len(results),
        "retrieve": dict(retrieve_params or {}),
    }
    if summary_path:
        os.makedirs(os.path.dirname(os.path.abspath(summary_path)), exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
    return summary


def _run_job(backend, job: dict, *, no_progress: bool, progress_interval: int) -> tuple[list, dict, float]:
    """在已加载的离线后端上执行一个参数组，避免重复读缓存和构建图适配器。"""
    required = {"id", "run_dir", "output", "summary", "beam_size", "lambda_val", "threshold", "eta"}
    missing = sorted(required - set(job))
    if missing:
        raise ValueError(f"批量评测任务缺少字段: {', '.join(missing)}")
    params = {
        **{key: job[key] for key in ("beam_size", "lambda_val", "threshold", "eta")},
        "step_score_mode": job.get("step_score_mode", "joint"),
        "penalty_mode": job.get("penalty_mode", "adaptive"),
    }
    validate_score_scheme(params["step_score_mode"], params["eta"])
    total = len(backend.bundle.samples)
    results = []
    interval = max(progress_interval, 1)
    started = time.perf_counter()
    with tqdm(total=total, desc=f"参数扫描 {job['id']}", unit="题", dynamic_ncols=True,
              leave=False, disable=no_progress) as progress:
        for sample_index in range(total):
            results.append(backend.retrieve(sample_index, **params))
            progress.update(1)
            if progress.n % interval == 0 or progress.n == total:
                update_progress(job["run_dir"], completed=progress.n, total=total, phase="参数扫描")
    write_results(results, job["output"])
    summary = evaluate_results(backend, results, job["summary"], retrieve_params=params)
    update_progress(job["run_dir"], completed=total, total=total, status="completed", phase="参数扫描")
    emit_event(job["run_dir"], "phase_end", phase="参数扫描", samples=total)
    return results, summary, time.perf_counter() - started


def _load_jobs(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        jobs = json.load(fh)
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("批量评测任务文件必须是非空 JSON 数组")
    if not all(isinstance(job, dict) for job in jobs):
        raise ValueError("批量评测任务必须全部为对象")
    return jobs


def run_jobs(args) -> None:
    """单进程批量评测；当前仅允许离线缓存，以保证所有任务共享同一后端。"""
    if args.backend != "offline":
        raise ValueError("批量评测当前仅支持 --backend offline")
    jobs = _load_jobs(args.jobs_file)
    fallback = os.path.dirname(os.path.abspath(args.jobs_file))
    run_dir = configure_runtime(
        args, command="路径检索批量评测", fallback_run_dir=fallback,
        manifest={"dataset": args.dataset, "backbone": args.backbone, "backend": args.backend,
                  "cache": args.cache, "jobs_file": os.path.abspath(args.jobs_file), "jobs": len(jobs)},
    )
    backend = build_backend(args)
    with tqdm(total=len(jobs), desc="参数扫描任务", unit="组", dynamic_ncols=True,
              disable=args.no_progress) as progress:
        for job in jobs:
            _results, summary, elapsed = _run_job(
                backend, job, no_progress=args.no_progress, progress_interval=args.progress_interval)
            progress.update(1)
            print(json.dumps(summary["backbone"]["overall"], ensure_ascii=False), flush=True)
            speed = len(_results) / elapsed if elapsed else 0.0
            print(f"[INFO] 参数扫描 {job['id']}：完成 {len(_results)} 条，"
                  f"耗时 {elapsed:.1f} 秒，{speed:.2f} 题/s", flush=True)
    update_progress(run_dir, completed=len(jobs), total=len(jobs), status="completed", phase="路径检索批量评测")
    emit_event(run_dir, "phase_end", phase="路径检索批量评测", jobs=len(jobs))


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.jobs_file:
        run_jobs(args)
        return
    started = time.perf_counter()
    backend, results = run_retrieval(args)
    retrieve_params = {
        "beam_size": args.beam_size,
        "lambda_val": args.lambda_val,
        "threshold": args.threshold,
        "eta": args.eta,
        "step_score_mode": args.step_score_mode,
        "penalty_mode": args.penalty_mode,
    }
    summary = evaluate_results(backend, results, args.summary, retrieve_params=retrieve_params)
    run_dir = getattr(args, "run_dir", "") or (os.path.dirname(os.path.abspath(args.summary)) if args.summary else "")
    update_progress(run_dir, completed=len(results), total=len(results), status="completed", phase="检索评测")
    emit_event(run_dir, "phase_end", phase="检索评测", samples=len(results))
    print(json.dumps(summary["backbone"]["overall"], ensure_ascii=False), flush=True)
    elapsed = time.perf_counter() - started
    speed = len(results) / elapsed if elapsed else 0.0
    print(f"[INFO] 检索评测完成 {len(results)} 条，耗时 {elapsed:.1f} 秒，{speed:.2f} 题/s", flush=True)


if __name__ == "__main__":
    main()
