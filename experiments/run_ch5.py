"""第五章渐进验证实验编排：正式评测、回放消融与敏感性。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from experiments.common import ROOT, require_fields, resolve_path, run_command
from kgqa.experiments import ExperimentPaths, load_confirmed_config, load_json_config
from kgqa.runtime import configure_runtime, emit_event, update_progress


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="第五章渐进验证实验编排")
    parser.add_argument("--dataset", choices=["webqsp", "metaqa"], required=True)
    parser.add_argument("--config", required=True, help="第五章实验配置 JSON")
    parser.add_argument("--profile", required=True, help="已人工确认的第三章检索配置 JSON")
    parser.add_argument("--phase", choices=["smoke", "benchmark", "replay_ablations", "sensitivity", "all"], default="all")
    parser.add_argument("--project_dir", default=str(ROOT))
    parser.add_argument("--dry_run", action="store_true", help="只展示命令和目标目录")
    return parser


def _agent_command(dataset: str, run: dict, output: Path, config: dict) -> list[str]:
    command = [
        sys.executable, "-m", "kgqa.agent.cli.eval_checked_batch", "--dataset", dataset,
        "--input", str(config["qa_file"]), "--output", str(output),
        "--limit", str(run.get("limit", 0)), "--beam_size", str(run["beam_size"]),
        "--lambda_val", str(run["lambda_val"]), "--alpha_final", str(run.get("alpha_final", 1.0)),
        "--path_threshold", str(run.get("path_threshold", 0.01)),
        "--batch_size", str(run["batch_size"]), "--check_mode", run["check_mode"],
        "--path_retrieve_url", config.get("path_retrieve_url", "http://localhost:8789"),
        "--llm_server_url", config.get("llm_server_url", "http://localhost:8788"),
        "--run_dir", str(output),
    ]
    for flag in run.get("flags", []):
        command.append(flag)
    return command


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_dir = Path(args.project_dir).resolve()
    config = load_json_config(args.config)
    profile = load_confirmed_config(args.profile)
    require_fields(config, "dataset", "qa_file", "runs")
    if config["dataset"] != args.dataset or profile["dataset"] != args.dataset:
        raise ValueError("第五章配置、检索配置和命令行数据集必须一致")
    paths = ExperimentPaths(project_dir)
    phases = [args.phase] if args.phase != "all" else ["smoke", "benchmark", "replay_ablations", "sensitivity"]
    for phase in phases:
        output = paths.ch5_dir(args.dataset, profile["config_id"], phase)
        configure_runtime(
            argparse.Namespace(run_dir=str(output), log_level="INFO"),
            command="第五章渐进验证",
            manifest={"config": str(Path(args.config).resolve()), "profile": str(Path(args.profile).resolve()), "phase": phase, "score_cache": profile.get("score_cache", {})},
        )
        if phase == "replay_ablations":
            benchmark = paths.ch5_dir(args.dataset, profile["config_id"], "benchmark")
            command = [
                sys.executable, "scripts/replay_ch5_ablation.py", "--canonical_dir", str(benchmark),
                "--output_root", str(output),
            ]
            run_command(command, output, dry_run=args.dry_run)
        else:
            for run in config["runs"].get(phase, []):
                run_output = output if len(config["runs"].get(phase, [])) == 1 else output / run["id"]
                run_command(_agent_command(args.dataset, run, run_output, config), run_output, dry_run=args.dry_run)
        update_progress(output, completed=1, total=1, status="completed", phase=phase)
        emit_event(output, "phase_end", phase=phase)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
