"""同一模型/adapter 的多份检索输入批量评测，避免重复加载模型。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from kgqa.pfit.eval import load_inference_model, run_eval
from kgqa.runtime import configure_runtime, emit_event, update_progress


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="pfit 多检索输入批量评测（模型仅加载一次）")
    parser.add_argument("--jobs_file", required=True, help="评测作业 JSON 文件")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--format", dest="fmt", required=True)
    parser.add_argument("--path_format", required=True)
    parser.add_argument("--entity_repr", required=True)
    parser.add_argument("--max_new_tokens", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--progress_interval", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    jobs = json.loads(Path(args.jobs_file).read_text(encoding="utf-8"))
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("jobs_file 必须是非空作业列表")
    model_obj, tokenizer = load_inference_model(
        model=args.model, max_seq_length=args.max_seq_length, adapter=args.adapter
    )
    for job in jobs:
        for key in ("input", "exp_dir", "run_dir", "manifest"):
            if key not in job:
                raise ValueError(f"评测作业缺少字段: {key}")
        run_dir = Path(job["run_dir"])
        configure_runtime(
            argparse.Namespace(run_dir=str(run_dir), log_level=args.log_level),
            command="第三章下游大模型问答", manifest=job["manifest"],
        )
        summary = run_eval(
            dataset=args.dataset, input_path=job["input"], exp_dir=job["exp_dir"],
            adapter=args.adapter, fmt=args.fmt, path_format=args.path_format,
            entity_repr=args.entity_repr, no_paths=bool(job.get("no_paths")),
            max_paths=int(job.get("max_paths", 0)),
            system_prompt_file=job.get("system_prompt_file"),
            limit=int(job.get("limit", 0)),
            model=args.model, max_seq_length=args.max_seq_length,
            max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
            show_progress=not args.no_progress, progress_interval=args.progress_interval,
            run_dir=str(run_dir), loaded_model=model_obj, loaded_tokenizer=tokenizer,
        )
        update_progress(run_dir, completed=1, total=1, status="completed", phase="下游大模型问答")
        emit_event(run_dir, "phase_end", condition=job.get("condition"), layer=job.get("layer"))
        print(json.dumps({"condition": job.get("condition"), "overall": summary.get("overall", summary)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
