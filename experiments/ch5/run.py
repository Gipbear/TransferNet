"""第五章渐进验证实验编排：正式评测、回放消融与敏感性。"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from experiments.common import ROOT, require_fields, run_command
from kgqa.experiments import ExperimentPaths, load_confirmed_config, load_json_config
from kgqa.runtime import configure_runtime, emit_event, update_progress


def _run_tag(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise argparse.ArgumentTypeError("run_tag 必须以字母或数字开头，且只能包含字母、数字、下划线、点和短横线")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="第五章渐进验证实验编排")
    parser.add_argument("--dataset", choices=["webqsp", "metaqa", "cwq"], required=True)
    parser.add_argument("--config", required=True, help="第五章实验配置 JSON")
    parser.add_argument("--profile", required=True, help="已人工确认的第三章检索配置 JSON")
    parser.add_argument("--phase", choices=["smoke", "benchmark", "replay_ablations", "sensitivity", "all"], default="all")
    parser.add_argument("--project_dir", default=str(ROOT))
    parser.add_argument("--run_tag", type=_run_tag, default=None, help="阶段目录下的隔离运行标识；不传沿用历史目录")
    parser.add_argument("--dry_run", action="store_true", help="只展示命令和目标目录")
    return parser


def _agent_command(dataset: str, run: dict, output: Path, config: dict) -> list[str]:
    if config.get("kind") == "ch5_workflow":
        return _workflow_command(dataset, run, output, config)
    if "alpha_final" in run:
        raise ValueError("第五章配置不接受已废弃字段 alpha_final，请改用 eta")
    command = [
        sys.executable, "-m", "kgqa.agent.cli.eval_checked_batch", "--dataset", dataset,
        "--input", str(config["qa_file"]), "--output", str(output),
        "--limit", str(run.get("limit", 0)), "--beam_size", str(run["beam_size"]),
        "--lambda_val", str(run["lambda_val"]), "--eta", str(run.get("eta", 1.0)),
        "--path_threshold", str(run.get("path_threshold", 0.01)),
        "--batch_size", str(run["batch_size"]), "--check_mode", run["check_mode"],
        "--path_retrieve_url", config.get("path_retrieve_url", "http://localhost:8789"),
        "--llm_server_url", config.get("llm_server_url", "http://localhost:8788"),
        "--run_dir", str(output),
    ]
    comparison_answers = config.get("cwq_comparison_answers")
    if comparison_answers:
        if dataset != "cwq":
            raise ValueError("cwq_comparison_answers 仅适用于 cwq 数据集")
        command.extend(["--cwq_comparison_answers", str(comparison_answers)])
    command.extend(run.get("flags", []))
    return command


def _workflow_command(dataset: str, run: dict, output: Path, config: dict) -> list[str]:
    if dataset != "cwq":
        raise ValueError("三路工作流当前仅支持 cwq")
    return [sys.executable, "-m", "kgqa.agent.cli.run_workflow",
            "--config", config["_config_path"], "--input", config["qa_file"],
            "--templates", config["templates"], "--cache_dir", config["cache_dir"],
            "--output", str(output), "--limit", str(run.get("limit", 0)),
            "--phase", run.get("workflow_phase", "full"), "--workers", str(run.get("workers", 4))]


def _workflow_scoring(run: dict, output: Path, config: dict) -> list[str]:
    command = [sys.executable, "-m", "kgqa.agent.cli.score_workflow", "--predictions",
               str(output / "predictions_no_gold.jsonl"), "--labels_source", config["labels_source"],
               "--output", str(output / "metrics.json")]
    if run.get("limit", 0):
        command.append("--allow_partial")
    return command


def _workflow_replay(benchmark: Path, output: Path, config: dict) -> list[str]:
    runs = config["runs"].get("benchmark", [])
    if len(runs) != 1:
        raise ValueError("工作流回放须指定唯一 benchmark 运行")
    predictions = benchmark / runs[0]["id"] / "predictions_no_gold.jsonl"
    return [sys.executable, "-m", "kgqa.agent.cli.replay_workflow", "--predictions", str(predictions),
            "--labels_source", config["labels_source"], "--output", str(output / "replay_metrics.json")]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_dir = Path(args.project_dir).resolve()
    config = load_json_config(args.config)
    config["_config_path"] = str(Path(args.config).resolve())
    profile = load_confirmed_config(args.profile)
    require_fields(config, "dataset", "qa_file", "runs")
    if config["dataset"] != args.dataset or profile["dataset"] != args.dataset:
        raise ValueError("第五章配置、检索配置和命令行数据集必须一致")
    paths = ExperimentPaths(project_dir)
    phases = [args.phase] if args.phase != "all" else ["smoke", "benchmark", "replay_ablations", "sensitivity"]
    for phase in phases:
        output = paths.ch5_dir(args.dataset, profile["config_id"], phase)
        if args.run_tag:
            output = output / args.run_tag
        configure_runtime(
            argparse.Namespace(run_dir=str(output), log_level="INFO"),
            command="第五章渐进验证",
            manifest={"config": str(Path(args.config).resolve()), "profile": str(Path(args.profile).resolve()), "phase": phase, "score_cache": profile.get("score_cache", {})},
        )
        if phase == "replay_ablations":
            benchmark = paths.ch5_dir(args.dataset, profile["config_id"], "benchmark")
            if args.run_tag:
                benchmark = benchmark / args.run_tag
            if config.get("kind") == "ch5_workflow":
                command = _workflow_replay(benchmark, output, config)
            else:
                command = [sys.executable, "scripts/replay_ch5_ablation.py", "--canonical_dir", str(benchmark),
                           "--output_root", str(output)]
            run_command(command, output, dry_run=args.dry_run)
        else:
            for run in config["runs"].get(phase, []):
                single_legacy = len(config["runs"].get(phase, [])) == 1 and config.get("kind") != "ch5_workflow"
                run_output = output if single_legacy else output / run["id"]
                run_command(_agent_command(args.dataset, run, run_output, config), run_output, dry_run=args.dry_run)
                if config.get("kind") == "ch5_workflow":
                    command = _workflow_scoring(run, run_output, config)
                    run_command(command, run_output, dry_run=args.dry_run)
        update_progress(output, completed=1, total=1, status="completed", phase=phase)
        emit_event(output, "phase_end", phase=phase)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
