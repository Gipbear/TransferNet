"""MetaQA P4：构造 3-hop 得分子缓存并汇总四组路径主对照。"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from experiments.common import ROOT, require_fields, resolve_path
from kgqa.experiments import ExperimentPaths, load_json_config
from kgqa.runtime import file_fingerprint


METHOD_LABELS = {
    "sp": "SP",
    "score": "得分引导候选路径",
    "fixed": "固定惩罚",
    "adaptive": "完整方法（TARRS）",
}


def filter_score_cache_by_hop(
    cache: dict[str, Any], *, hop: int, split: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """按数据集标注 hop 筛选统一 score cache，并保持源顺序。"""
    if not isinstance(cache, dict) or not isinstance(cache.get("meta"), dict):
        raise ValueError("score cache 缺少 meta 对象")
    samples = cache.get("samples")
    if not isinstance(samples, list):
        raise ValueError("score cache 缺少 samples 列表")

    hop_counts: Counter[int] = Counter()
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or sample.get("hop") is None:
            raise ValueError(f"score cache 样本 {index} 缺少 hop 标签")
        try:
            hop_counts[int(sample["hop"])] += 1
        except (TypeError, ValueError) as exc:
            raise ValueError(f"score cache 样本 {index} 的 hop 非法: {sample.get('hop')!r}") from exc

    selected = [sample for sample in samples if int(sample["hop"]) == hop]
    if not selected:
        raise ValueError(f"score cache 中没有 hop={hop} 的样本")

    source_meta = cache["meta"]
    filtered = dict(cache)
    filtered["meta"] = {
        **source_meta,
        "split": split,
        "num_samples": len(selected),
        "source_split": source_meta.get("split", ""),
        "hop_filter": hop,
    }
    filtered["samples"] = selected
    manifest = {
        "schema_version": 1,
        "source_split": source_meta.get("split", ""),
        "output_split": split,
        "source_samples": len(samples),
        "source_hop_counts": {str(key): hop_counts[key] for key in sorted(hop_counts)},
        "selected_hop": hop,
        "selected_samples": len(selected),
        "order_preserved": True,
    }
    return filtered, manifest


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def prepare_p4_cache(config: dict[str, Any], project_dir: Path) -> dict[str, Any]:
    p4 = config.get("p4")
    if not isinstance(p4, dict):
        raise ValueError("MetaQA P4 配置缺少 p4 对象")
    require_fields(p4, "dataset_hop", "expected_samples", "source_cache", "prepared_cache", "manifest")
    hop = int(p4["dataset_hop"])
    split = str(config.get("selection_split", ""))
    if not split:
        raise ValueError("MetaQA P4 配置缺少 selection_split")

    source_path = resolve_path(project_dir, p4["source_cache"])
    output_path = resolve_path(project_dir, p4["prepared_cache"])
    manifest_path = resolve_path(project_dir, p4["manifest"])
    cache = torch.load(source_path, map_location="cpu", weights_only=False)
    filtered, manifest = filter_score_cache_by_hop(cache, hop=hop, split=split)
    if len(filtered["samples"]) != int(p4["expected_samples"]):
        raise ValueError(
            f"hop={hop} 样本数不符合配置: "
            f"{len(filtered['samples'])} != {p4['expected_samples']}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(filtered, temporary)
    os.replace(temporary, output_path)
    manifest.update({
        "source_cache": file_fingerprint(source_path),
        "prepared_cache": file_fingerprint(output_path),
    })
    _write_json(manifest_path, manifest)
    return manifest


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} 不是 JSON 对象")
            rows.append(value)
    return rows


def _validate_path_format(method: str, rows: list[dict[str, Any]]) -> None:
    for row_index, row in enumerate(rows):
        paths = row.get("mmr_reason_paths")
        if not isinstance(paths, list):
            raise ValueError(f"{method} 样本 {row_index} 的 mmr_reason_paths 不是列表")
        for path_index, path in enumerate(paths):
            if not isinstance(path, dict) or not isinstance(path.get("path"), list):
                raise ValueError(f"{method} 样本 {row_index} 路径 {path_index} 格式非法")
            for edge in path["path"]:
                if not isinstance(edge, (list, tuple)) or len(edge) != 3:
                    raise ValueError(f"{method} 样本 {row_index} 路径 {path_index} 三元组格式非法")


def build_p4_report(
    result_paths: dict[str, Path],
    summary_paths: dict[str, Path],
    *,
    expected_samples: int,
    dataset_hop: int,
) -> dict[str, Any]:
    """校验四组结果严格对齐，并形成论文表格所需的机器可读汇总。"""
    if set(result_paths) != set(METHOD_LABELS) or set(summary_paths) != set(METHOD_LABELS):
        raise ValueError(f"P4 必须提供四组方法: {', '.join(METHOD_LABELS)}")

    rows_by_method = {method: _load_jsonl(result_paths[method]) for method in METHOD_LABELS}
    for method, rows in rows_by_method.items():
        if len(rows) != expected_samples:
            raise ValueError(f"{method} 样本数不符: {len(rows)} != {expected_samples}")
        _validate_path_format(method, rows)

    alignment_fields = ["sample_index", "question", "golden"]
    baseline = [tuple(row.get(field) if field != "golden" else tuple(row.get(field, []))
                      for field in alignment_fields)
                for row in rows_by_method["sp"]]
    for method in tuple(METHOD_LABELS)[1:]:
        current = [tuple(row.get(field) if field != "golden" else tuple(row.get(field, []))
                         for field in alignment_fields)
                   for row in rows_by_method[method]]
        if current != baseline:
            raise ValueError(f"{method} 与 sp 的题目顺序或 golden 不一致")

    methods: dict[str, Any] = {}
    for method in METHOD_LABELS:
        summary = json.loads(summary_paths[method].read_text(encoding="utf-8"))
        if summary.get("n") != expected_samples:
            raise ValueError(f"{method} summary 样本数不符: {summary.get('n')} != {expected_samples}")
        overall = summary.get("path", {}).get("overall", {})
        if overall.get("n") != expected_samples:
            raise ValueError(
                f"{method} path summary 样本数不符: {overall.get('n')} != {expected_samples}"
            )
        rows = rows_by_method[method]
        methods[method] = {
            "label": METHOD_LABELS[method],
            "n": expected_samples,
            "answer_hit_at_k": overall.get("answer_hit"),
            "top1_hit": overall.get("top1_hit"),
            "precision": overall.get("precision"),
            "recall": overall.get("recall"),
            "f1": overall.get("f1"),
            "relation_jaccard_diversity": overall.get("relation_jaccard_diversity"),
            "average_paths": round(
                sum(len(row["mmr_reason_paths"]) for row in rows) / expected_samples,
                4,
            ),
            "result": str(result_paths[method]),
            "summary": str(summary_paths[method]),
        }

    return {
        "schema_version": 1,
        "dataset": "metaqa",
        "dataset_hop_filter": dataset_hop,
        "n": expected_samples,
        "alignment": {
            "passed": True,
            "fields": alignment_fields,
            "samples": expected_samples,
            "path_format": "mmr_reason_paths[].path[][subject, relation, object]",
        },
        "methods": methods,
    }


def _p4_artifacts(
    config: dict[str, Any], project_dir: Path,
) -> tuple[dict[str, Path], dict[str, Path]]:
    paths = ExperimentPaths(project_dir)
    dataset = config["dataset"]
    backbone = config["backbone"]
    config_id = config["config_id"]
    split = config["selection_split"]
    shortest_id = config["shortest_path_baseline"]["id"]
    shortest_dir = paths.ch3_shortest_path_dir(dataset, backbone, config_id) / shortest_id
    penalty_dir = paths.ch3_penalty_ablation_dir(dataset, backbone, config_id)
    result_paths = {
        "sp": shortest_dir / f"{split}.jsonl",
        "score": penalty_dir / "none" / f"{split}.jsonl",
        "fixed": penalty_dir / "fixed" / f"{split}.jsonl",
        "adaptive": penalty_dir / "adaptive" / f"{split}.jsonl",
    }
    summary_paths = {
        method: path.with_name(f"{split}_summary.json")
        for method, path in result_paths.items()
    }
    return result_paths, summary_paths


def write_p4_report(config: dict[str, Any], project_dir: Path) -> tuple[Path, dict[str, Any]]:
    p4 = config.get("p4")
    if not isinstance(p4, dict):
        raise ValueError("MetaQA P4 配置缺少 p4 对象")
    require_fields(p4, "dataset_hop", "expected_samples", "manifest", "report")
    manifest_path = resolve_path(project_dir, p4["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("selected_samples") != int(p4["expected_samples"]):
        raise ValueError("3-hop 子缓存 manifest 与 P4 配置的样本数不一致")
    if manifest.get("selected_hop") != int(p4["dataset_hop"]):
        raise ValueError("3-hop 子缓存 manifest 与 P4 配置的 hop 不一致")

    result_paths, summary_paths = _p4_artifacts(config, project_dir)
    report = build_p4_report(
        result_paths,
        summary_paths,
        expected_samples=int(p4["expected_samples"]),
        dataset_hop=int(p4["dataset_hop"]),
    )
    report["subset_manifest"] = str(manifest_path)
    report_path = resolve_path(project_dir, p4["report"])
    _write_json(report_path, report)
    return report_path, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MetaQA P4 3-hop 路径主对照准备与汇总")
    parser.add_argument("--phase", choices=["prepare", "report"], required=True)
    parser.add_argument(
        "--config",
        default=str(ROOT / "experiments/configs/ch3/metaqa_transfernet_v1_p4.json"),
    )
    parser.add_argument("--project_dir", default=str(ROOT))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_dir = Path(args.project_dir).resolve()
    config = load_json_config(args.config)
    require_fields(config, "dataset", "backbone", "config_id", "selection_split", "p4")
    if config["dataset"] != "metaqa":
        raise ValueError("MetaQA P4 配置的 dataset 必须是 metaqa")
    if args.phase == "prepare":
        manifest = prepare_p4_cache(config, project_dir)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    else:
        report_path, report = write_p4_report(config, project_dir)
        print(f"P4 汇总已写入: {report_path}")
        print(json.dumps(report["methods"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
