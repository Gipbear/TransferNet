"""通过固定服务和冻结配置运行 CWQ 初答或完整三路工作流。"""

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from kgqa.agent.workflow.config import WorkflowConfig
from kgqa.agent.workflow.models import RuntimeQuestion
from kgqa.agent.workflow.persistence import ArtifactStore
from kgqa.agent.workflow.pfit import PfitAnswerer
from kgqa.agent.workflow.resources import EntityNameResource
from kgqa.agent.workflow.run import WorkflowRunner
from kgqa.agent.workflow.templates import TemplateLibrary
from kgqa.runtime import file_fingerprint


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CWQ 三路问答工作流")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--templates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--phase", choices=["baseline", "full"], default="full")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--wait_initial", action="store_true", help="消费同配置初答阶段正在产生的缓存")
    args = parser.parse_args(argv)
    os.environ["NO_PROXY"] = "localhost,127.0.0.1"
    if args.limit < 0 or args.workers < 1:
        parser.error("limit 须非负，workers 须为正数")
    output = Path(args.output)
    with ArtifactStore.exclusive(output / ".run.lock"):
        execute(args, output)


def execute(args: argparse.Namespace, output: Path) -> None:
    raw = json.loads(Path(args.config).read_text())
    config = WorkflowConfig.from_dict(raw["workflow"])
    names = EntityNameResource.load(raw.get("entity_names_file"))
    if args.phase == "baseline":
        config = replace(config, comparison_enabled=False, conjunction_enabled=False, rescue_enabled=False)
    with Path(args.input).open() as handle:
        questions = [RuntimeQuestion.from_record(json.loads(line)) for line in handle if line.strip()]
    if args.limit:
        questions = questions[:args.limit]
    output.mkdir(parents=True, exist_ok=True)
    fingerprints = {name: file_fingerprint(path) for name, path in {
        "input": args.input, "templates": args.templates, "config": args.config,
        "adapter": str(Path(raw["adapter"]) / "adapter_model.safetensors"),
    }.items()}
    fingerprints["entity_names"] = names.fingerprint
    source = {str(path): file_fingerprint(path)["sha256"]
              for path in sorted(Path("kgqa/agent/workflow").rglob("*.py"))}
    frozen = {"schema_version": 3, "phase": args.phase, "workflow": config.to_dict(),
              "fingerprints": fingerprints, "limit": args.limit, "source_sha256": source}
    manifest = output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != frozen:
        raise ValueError("运行配置或输入发生变化，请使用新目录")
    ArtifactStore.atomic_text(manifest, json.dumps(frozen, ensure_ascii=False, indent=2) + "\n")
    ArtifactStore.atomic_text(output / "config.json", json.dumps(raw, ensure_ascii=False, indent=2) + "\n")
    answerer = PfitAnswerer(config, raw["model"], raw["adapter"], Path(args.cache_dir) / "pfit",
                           raw.get("llm_url", "http://127.0.0.1:8788"))
    library = TemplateLibrary.load(args.templates)
    runner = WorkflowRunner(config, library, answerer, output, Path(args.cache_dir),
                            raw.get("graph_url", "http://127.0.0.1:8791"), args.wait_initial, names=names.names)
    report = runner.run(questions, args.workers)
    report["workers"] = args.workers
    report["wait_initial"] = args.wait_initial
    ArtifactStore.atomic_text(output / "run_summary.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
