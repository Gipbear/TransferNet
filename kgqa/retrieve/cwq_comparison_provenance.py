"""记录 CWQ 比较题推断输入来源并验证未使用测试标签。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_provenance(args: argparse.Namespace, total: int, selected: int) -> None:
    sources = {
        name: str(getattr(args, name))
        for name in (
            "train", "train_retrieval", "test_retrieval", "rog_qa", "mid_qa",
            "rog_entities", "mid_entities",
        )
    }
    if args.dev is not None:
        sources["dev"] = str(args.dev)
    training_source = "train/dev" if args.dev is not None else "train"
    manifest = {
        "test_count": total,
        "predicted_comparison_count": selected,
        "source_sha256": {name: _sha256(Path(path)) for name, path in sources.items()},
        "field_sources": {
            "predicted_type": f"{training_source} compositionality_type labels + test_retrieval.question",
            "attribute_chain": f"{training_source} SPARQL labels + test_retrieval.question",
            "operator": f"{training_source} SPARQL labels + test_retrieval.question",
            "threshold": "test_retrieval.question",
            "candidates": "test_retrieval.mmr_reason_paths tail names",
            "candidate_mids": "paired rog_qa/mid_qa subgraph tuples + entity vocabularies",
            "main_relation": "train SPARQL + test question/topic MID/candidate MID + QLever Freebase",
            "path_support": "train_retrieval gold-tail paths + test_retrieval question/topics/paths",
            "property_values": "QLever Freebase by predicted chain and candidate MID",
        },
        "test_fields_excluded": ["answers", "golden", "hit_any", "prediction", "sparql"],
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
