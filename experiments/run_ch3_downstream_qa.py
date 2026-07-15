"""第三章多检索路径下游大模型问答编排。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from experiments.ch3_downstream_qa import (
    CONDITION_IDS,
    condition_by_id,
    load_downstream_config,
    resolve_fixed_adapter,
    validate_condition_inputs,
)
from experiments.ch3_downstream_report import write_report
from experiments.common import ROOT, resolve_path, run_command
from kgqa.experiments import ExperimentPaths
from kgqa.runtime import file_fingerprint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="第三章多检索路径下游大模型问答编排")
    parser.add_argument("--dataset", choices=["webqsp"], required=True, help="数据集")
    parser.add_argument(
        "--config", default=str(ROOT / "experiments/configs/ch3/webqsp_transfernet_v1_downstream_qa.json"),
        help="第三章下游 QA 配置 JSON",
    )
    parser.add_argument("--layer", choices=["base_zeroshot", "fixed_pfit_adapter"], default="base_zeroshot")
    parser.add_argument("--condition", choices=["all", *CONDITION_IDS], default="all")
    parser.add_argument("--phase", choices=["validate", "eval", "report", "all"], default="all")
    parser.add_argument("--smoke_size", type=int, default=0, help="按 hop 分层抽取的共同冒烟样本数；0 表示全量")
    parser.add_argument("--project_dir", default=str(ROOT), help="项目根目录")
    parser.add_argument("--dry_run", action="store_true", help="只校验输入并展示命令，不加载模型")
    parser.add_argument("--no_progress", action="store_true", help="关闭模型评测进度条")
    parser.add_argument("--progress_interval", type=int, default=50, help="模型评测进度更新间隔")
    parser.add_argument("--log_level", default="INFO", help="日志级别")
    return parser


def _selected_conditions(requested: str) -> tuple[str, ...]:
    return CONDITION_IDS if requested == "all" else (requested,)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_dir = Path(args.project_dir).resolve()
    config = load_downstream_config(args.config, project_dir)
    if args.dataset != config["dataset"]:
        raise ValueError("命令行数据集与下游 QA 配置不一致")
    input_info = validate_condition_inputs(config, project_dir)
    paths = ExperimentPaths(project_dir)
    root_dir = paths.ch3_downstream_qa_dir(config["dataset"], config["backbone"], config["config_id"])

    adapter_id: str | None = None
    adapter_path: Path | None = None
    adapter_fingerprint: dict[str, Any] | None = None
    if args.layer == "fixed_pfit_adapter":
        adapter_id, adapter_path, adapter_fingerprint = resolve_fixed_adapter(config, project_dir)
        layer_base_dir = root_dir / "fixed_pfit_adapter" / adapter_id
    else:
        layer_base_dir = root_dir / "base_zeroshot"

    input_paths = {
        condition["id"]: resolve_path(project_dir, condition["input"])
        for condition in config["conditions"]
    }
    run_name = "full"
    if args.smoke_size:
        run_name = f"smoke_{args.smoke_size}"
        smoke_dir = root_dir / "smoke_inputs" / run_name
        if args.dry_run:
            print(f"[演练] 将按 hop 分层生成 {args.smoke_size} 条共同冒烟输入到 {smoke_dir}")
            input_paths = {condition_id: smoke_dir / f"{condition_id}.jsonl" for condition_id in CONDITION_IDS}
        else:
            from experiments.ch3_downstream_qa import write_stratified_smoke_inputs
            input_paths = write_stratified_smoke_inputs(config, project_dir, smoke_dir, args.smoke_size)
            input_info = {
                condition_id: {
                    "input": file_fingerprint(path),
                    "samples": args.smoke_size,
                    "qa_signature": "由完整输入对齐校验后按 hop 共同抽样",
                }
                for condition_id, path in input_paths.items()
            }
    layer_dir = layer_base_dir / run_name

    if args.phase in {"validate", "all"}:
        print(json.dumps({"inputs": input_info, "layer": args.layer}, ensure_ascii=False, indent=2))
    if args.phase in {"eval", "all"}:
        jobs: list[dict[str, Any]] = []
        for condition_id in _selected_conditions(args.condition):
            condition = condition_by_id(config, condition_id)
            input_path = input_paths[condition_id]
            run_dir = layer_dir / condition_id
            manifest = {
                "config": config["_config_path"], "profile": config["_profile_path"],
                "layer": args.layer, "condition": condition_id, "method": condition["method"],
                "input": input_info[condition_id]["input"], "qa_signature": input_info[condition_id]["qa_signature"],
                "evaluation": config["evaluation"], "adapter": adapter_fingerprint,
            }
            jobs.append({
                "condition": condition_id, "layer": args.layer, "input": str(input_path),
                "exp_dir": str(run_dir), "run_dir": str(run_dir),
                "no_paths": condition_id == "no_path", "manifest": manifest,
            })
        batch_dir = layer_dir / "batch"
        jobs_file = batch_dir / "jobs.json"
        if not args.dry_run:
            jobs_file.parent.mkdir(parents=True, exist_ok=True)
            jobs_file.write_text(json.dumps(jobs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            for job in jobs:
                console_path = Path(job["run_dir"]) / "logs" / "console.log"
                console_path.parent.mkdir(parents=True, exist_ok=True)
                console_path.write_text(
                    f"[信息] 此条件由模型复用批处理执行；完整控制台输出见 {batch_dir / 'logs/console.log'}\n",
                    encoding="utf-8",
                )
        command = [
            sys.executable, "-m", "kgqa.pfit.eval_batch", "--jobs_file", str(jobs_file),
            "--dataset", args.dataset, "--model", config["evaluation"]["model"],
            "--format", config["evaluation"]["format"], "--path_format", config["evaluation"]["path_format"],
            "--entity_repr", config["evaluation"]["entity_repr"],
            "--max_new_tokens", str(config["evaluation"]["max_new_tokens"]),
            "--batch_size", str(config["evaluation"]["batch_size"]), "--log_level", args.log_level,
            "--progress_interval", str(args.progress_interval),
        ]
        if adapter_path is not None:
            command.extend(["--adapter", str(adapter_path)])
        if args.no_progress:
            command.append("--no_progress")
        run_command(command, batch_dir, dry_run=args.dry_run)
    if args.phase in {"report", "all"}:
        if args.condition != "all":
            raise ValueError("汇总报告必须读取全部五个条件，请使用 --condition all")
        if args.dry_run:
            print(f"[演练] 将汇总 {layer_dir} 到 {root_dir / 'reports' / args.layer / run_name}")
        else:
            report_dir = root_dir / "reports" / args.layer / run_name
            write_report(
                config=config, input_info=input_info, input_paths=input_paths,
                layer_dir=layer_dir, report_dir=report_dir,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
