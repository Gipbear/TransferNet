"""第三章检索实验编排：top-k 饱和性与人工确认检索配置。"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

from experiments.common import ROOT, require_fields, resolve_path, run_command
from kgqa.experiments import ExperimentPaths, load_confirmed_config, load_json_config
from kgqa.runtime import configure_runtime, emit_event, update_progress


def _default_config(project_dir: Path, dataset: str, backbone: str) -> Path:
    return project_dir / "experiments" / "configs" / "ch3" / f"{dataset}_{backbone}_v1.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="第三章检索实验：top-k 饱和性与检索参数扫描")
    parser.add_argument("--dataset", choices=["webqsp", "metaqa", "cwq"], required=True)
    parser.add_argument("--backbone", default="transfernet", choices=["transfernet", "rearev"])
    parser.add_argument("--config", default="", help="版本化检索配置 JSON；默认按数据集与基础检索模型选择")
    parser.add_argument("--phase", choices=["scores", "scan", "publish", "all"], default="all")
    parser.add_argument("--project_dir", default=str(ROOT), help="项目根目录，仅供实验编排定位配置与产物")
    parser.add_argument("--dry_run", action="store_true", help="只展示命令和目标目录，不执行模型")
    return parser


def _score_source(config: dict[str, Any], split: str) -> dict[str, str]:
    source = config.get("score_source", {})
    split_source = source.get("splits", {}).get(split, {})
    required = {"ckpt": source.get("ckpt", ""), "input_dir": source.get("input_dir", ""), "qa_file": split_source.get("qa_file", "")}
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"得分缓存配置缺少 {split} 的字段: {', '.join(missing)}")
    return required


def _retrieve_args(config: dict[str, Any], override: dict[str, Any]) -> list[str]:
    params = {**config["retrieve"], **override}
    return [
        "--beam_size", str(params["beam_size"]),
        "--lambda_val", str(params["lambda_val"]),
        "--threshold", str(params["threshold"]),
        "--alpha_final", str(params["alpha_final"]),
    ]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_dir = Path(args.project_dir).resolve()
    config_path = Path(args.config) if args.config else _default_config(project_dir, args.dataset, args.backbone)
    config = load_json_config(config_path)
    require_fields(config, "dataset", "backbone", "config_id", "topk", "retrieve")
    if config["dataset"] != args.dataset or config["backbone"] != args.backbone:
        raise ValueError("命令行数据集/基础检索模型与配置不一致")
    if args.backbone == "rearev" and args.phase in {"scores", "all"}:
        raise ValueError("ReaRev 当前只消费既有离线得分缓存，不能生成新的 score 缓存")

    paths = ExperimentPaths(project_dir)
    saturation_dir = paths.ch3_saturation_dir(args.dataset, args.backbone, config["config_id"])
    profile_dir = paths.ch3_profile_dir(args.dataset, args.backbone, config["config_id"])

    if args.phase in {"scores", "all"}:
        for topk in config.get("topk_candidates", [100, 250, 500, 1000]):
            for split in config.get("score_source", {}).get("splits", {}):
                source = _score_source(config, split)
                score_id = f"topk{topk}_{split}"
                run_dir = saturation_dir / score_id
                cache_path = paths.score_dir(args.dataset, args.backbone, score_id) / f"{split}.pt"
                configure_runtime(
                    argparse.Namespace(run_dir=str(run_dir), log_level="INFO"),
                    command="第三章 top-k 饱和性得分缓存",
                    manifest={"config_path": str(config_path), "topk": topk, "split": split, "output": str(cache_path)},
                )
                command = [
                    sys.executable, "-m", "kgqa.retrieve.cli.dump_scores",
                    "--dataset", args.dataset, "--ckpt", str(resolve_path(project_dir, source["ckpt"])),
                    "--input_dir", str(resolve_path(project_dir, source["input_dir"])),
                    "--qa_file", str(resolve_path(project_dir, source["qa_file"])),
                    "--split", split, "--topk", str(topk), "--output", str(cache_path),
                    "--run_dir", str(run_dir),
                ]
                run_command(command, run_dir, dry_run=args.dry_run)
                update_progress(run_dir, completed=1, total=1, status="completed", phase="得分缓存")
                emit_event(run_dir, "phase_end", phase="得分缓存")

    if args.phase in {"scan", "all"}:
        scan_items = config.get("parameter_scan", [])
        if not scan_items:
            raise ValueError("检索配置没有 parameter_scan，无法执行参数扫描")
        for item in scan_items:
            scan_id = item["id"]
            for split in config.get("score_source", {}).get("splits", {}):
                score_id = f"topk{config['topk']}_{split}"
                cache_path = paths.score_dir(args.dataset, args.backbone, score_id) / f"{split}.pt"
                run_dir = saturation_dir / "parameter_scan" / scan_id / split
                output = profile_dir / "candidates" / scan_id / f"{split}.jsonl"
                summary = profile_dir / "candidates" / scan_id / f"{split}_summary.json"
                source = _score_source(config, split)
                configure_runtime(
                    argparse.Namespace(run_dir=str(run_dir), log_level="INFO"),
                    command="第三章检索参数扫描",
                    manifest={"config_path": str(config_path), "candidate": scan_id, "split": split, "cache": str(cache_path)},
                )
                command = [
                    sys.executable, "-m", "kgqa.retrieve.cli.eval", "--dataset", args.dataset,
                    "--backend", "offline", "--cache", str(cache_path),
                    "--input_dir", str(resolve_path(project_dir, source["input_dir"])),
                    "--output", str(output), "--summary", str(summary), "--run_dir", str(run_dir),
                    *_retrieve_args(config, item.get("retrieve", {})),
                ]
                run_command(command, run_dir, dry_run=args.dry_run)
                update_progress(run_dir, completed=1, total=1, status="completed", phase="参数扫描")
                emit_event(run_dir, "phase_end", phase="参数扫描")

    if args.phase == "publish":
        # 人工确认后仅发布已选候选产物；不复制未确认的测试集候选。
        confirmed = load_confirmed_config(config_path)
        candidate_id = confirmed.get("selected_candidate")
        if not candidate_id:
            raise ValueError("已确认检索配置缺少 selected_candidate，无法发布正式检索结果")
        run_dir = profile_dir / "publish"
        configure_runtime(
            argparse.Namespace(run_dir=str(run_dir), log_level="INFO"),
            command="发布已确认检索配置",
            manifest={"config_path": str(config_path), "selected_candidate": candidate_id},
        )
        for split in confirmed.get("score_source", {}).get("splits", {}):
            source = profile_dir / "candidates" / candidate_id / f"{split}.jsonl"
            target = profile_dir / f"{split}.jsonl"
            if not source.is_file() and not args.dry_run:
                raise ValueError(f"已选候选缺少 {split} 检索结果: {source}")
            if args.dry_run:
                print(f"[演练] 发布 {source} → {target}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        if not args.dry_run:
            shutil.copy2(config_path, profile_dir / "confirmed_config.json")
        update_progress(run_dir, completed=1, total=1, status="completed", phase="发布已确认检索配置")
        emit_event(run_dir, "phase_end", phase="发布已确认检索配置", candidate=candidate_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
