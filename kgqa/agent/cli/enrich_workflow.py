"""验证本次权重哈希后补全早期轨迹，并可迁移同提示真实生成缓存。"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from kgqa.agent.workflow.config import WorkflowConfig
from kgqa.agent.workflow.persistence import ArtifactStore
from kgqa.agent.workflow.pfit import PfitPromptRenderer
from kgqa.agent.workflow.trace import CachedTraceReader
from kgqa.runtime import file_fingerprint


def load_reader(args: argparse.Namespace) -> CachedTraceReader:
    raw = json.loads(Path(args.config).read_text())
    frozen = json.loads(Path(args.manifest).read_text())
    adapter = str(Path(raw["adapter"]).resolve())
    fingerprint = file_fingerprint(Path(adapter) / "adapter_model.safetensors")
    if fingerprint["sha256"] != frozen["fingerprints"]["adapter"]["sha256"]:
        raise ValueError("权重与实际生成时哈希不一致，禁止导入缓存")
    if file_fingerprint(args.config)["sha256"] != frozen["fingerprints"]["config"]["sha256"]:
        raise ValueError("配置与实际生成时哈希不一致")
    tokenizer = AutoTokenizer.from_pretrained(raw["model"], local_files_only=True)
    config = WorkflowConfig.from_dict(raw["workflow"])
    renderer = PfitPromptRenderer(config, tokenizer)
    return CachedTraceReader(renderer, str(Path(raw["model"]).resolve()), adapter,
                             config.max_new_tokens, Path(args.cache_dir) / "pfit", fingerprint["sha256"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="补全实际生成轨迹元数据")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--import_cache", action="store_true")
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("limit 必须非负")
    reader = load_reader(args)
    rows = []
    with Path(args.predictions).open() as handle:
        for line in handle:
            rows.append(reader.enrich(json.loads(line), args.import_cache))
            if args.limit and len(rows) == args.limit:
                break
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    output = Path(args.output)
    ArtifactStore.atomic_text(output, content)
    audit = {"n": len(rows), "predictions_unchanged": True, "imported_cache": args.import_cache,
             "source": file_fingerprint(args.predictions), "output": file_fingerprint(output),
             "scope": "复用本次实际生成，不产生新预测；仅补齐实际提示路径和调用成本"}
    ArtifactStore.atomic_text(output.with_suffix(".audit.json"), json.dumps(audit, ensure_ascii=False, indent=2))
    print(json.dumps(audit, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
