"""第四章路径监督微调实验编排。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from experiments.common import ROOT, require_fields, resolve_path, run_command
from kgqa.experiments import ExperimentPaths, load_confirmed_config, load_json_config
from kgqa.runtime import configure_runtime, emit_event, update_progress


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="第四章路径监督微调实验编排")
    parser.add_argument("--dataset", choices=["webqsp", "metaqa", "cwq"], required=True)
    parser.add_argument("--config", required=True, help="第四章实验矩阵 JSON")
    parser.add_argument("--profile", required=True, help="已人工确认的第三章检索配置 JSON")
    parser.add_argument("--experiment", default="all", help="实验编号；all 表示矩阵中的全部实验")
    parser.add_argument("--phase", choices=["build", "train", "eval", "all"], default="all")
    parser.add_argument("--seed", type=int, default=None, help="只运行指定随机种子")
    parser.add_argument("--project_dir", default=str(ROOT))
    parser.add_argument("--dry_run", action="store_true", help="只展示命令和目标目录")
    return parser


def _selected_entries(config: dict[str, Any], requested: str) -> list[dict[str, Any]]:
    entries = config.get("experiments", [])
    if requested == "all":
        return entries
    matched = [entry for entry in entries if entry.get("id") == requested]
    if not matched:
        raise ValueError(f"第四章矩阵中未找到实验: {requested}")
    return matched


def _build_command(dataset: str, entry: dict[str, Any], train_input: Path, run_dir: Path, seed: int) -> list[str]:
    return [
        sys.executable, "-m", "kgqa.pfit.build", "--dataset", dataset,
        "--input", str(train_input), "--exp_dir", str(run_dir), "--seed", str(seed),
        *entry.get("build_args", []), "--run_dir", str(run_dir),
    ]


def _eval_command(
        dataset: str, entry: dict[str, Any], test_input: Path, run_dir: Path,
        adapter: str | None, seed: int) -> list[str]:
    command = [
        sys.executable, "-m", "kgqa.pfit.eval", "--dataset", dataset,
        "--input", str(test_input), "--exp_dir", str(run_dir), "--seed", str(seed),
        *entry.get("eval_args", []), "--run_dir", str(run_dir),
    ]
    if adapter:
        command.extend(["--adapter", adapter])
    return command


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_dir = Path(args.project_dir).resolve()
    config = load_json_config(args.config)
    profile = load_confirmed_config(args.profile)
    require_fields(config, "dataset", "config_id", "experiments")
    if config["dataset"] != args.dataset or profile["dataset"] != args.dataset:
        raise ValueError("第四章配置、检索配置和命令行数据集必须一致")
    paths = ExperimentPaths(project_dir)
    profile_dir = paths.ch3_profile_dir(profile["dataset"], profile["backbone"], profile["config_id"])
    for entry in _selected_entries(config, args.experiment):
        train_input = profile_dir / entry.get("train_file", config.get("train_file", "train.jsonl"))
        test_input = profile_dir / entry.get("test_file", config.get("test_file", "test.jsonl"))
        for seed in entry.get("seeds", [17]):
            if args.seed is not None and seed != args.seed:
                continue
            run_dir = paths.ch4_run_dir(args.dataset, profile["config_id"], entry["id"], seed)
            configure_runtime(
                argparse.Namespace(run_dir=str(run_dir), log_level="INFO"),
                command="第四章路径监督微调",
                manifest={"matrix_config": str(Path(args.config).resolve()), "profile_config": str(Path(args.profile).resolve()), "experiment": entry["id"], "seed": seed},
            )
            if entry.get("mode") == "train" and args.phase in {"build", "all"}:
                run_command(_build_command(args.dataset, entry, train_input, run_dir, seed), run_dir, dry_run=args.dry_run)
            if entry.get("mode") == "train" and args.phase in {"train", "all"}:
                train_command = [
                    sys.executable, "-m", "kgqa.pfit.train", "--exp_dir", str(run_dir), "--seed", str(seed),
                    *entry.get("train_args", []), "--run_dir", str(run_dir),
                ]
                run_command(train_command, run_dir, dry_run=args.dry_run)
            if args.phase in {"eval", "all"}:
                adapter = None
                if entry.get("adapter_from"):
                    source = paths.ch4_run_dir(args.dataset, profile["config_id"], entry["adapter_from"], seed)
                    adapter = str(source / "adapter")
                elif entry.get("mode") == "train":
                    adapter = str(run_dir / "adapter")
                command = _eval_command(args.dataset, entry, test_input, run_dir, adapter, seed)
                run_command(command, run_dir, dry_run=args.dry_run)
            update_progress(run_dir, completed=1, total=1, status="completed", phase="路径监督微调")
            emit_event(run_dir, "phase_end", phase="路径监督微调")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
