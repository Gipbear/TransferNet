"""独立评分已落盘输出的模块关闭回放，不触发模型或知识库。"""

import argparse
import json
from pathlib import Path

from kgqa.agent.workflow.ablations import WorkflowAblations
from kgqa.agent.workflow.persistence import ArtifactStore
from kgqa.agent.workflow.scoring import WorkflowScorer
from kgqa.runtime import file_fingerprint


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CWQ 工作流模块关闭回放")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--labels_source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow_partial", action="store_true")
    args = parser.parse_args(argv)
    records = WorkflowScorer._read(Path(args.predictions))
    labels = {r["sample_index"]: r for r in WorkflowScorer._read(Path(args.labels_source))}
    if not args.allow_partial and len(records) != len(labels):
        raise ValueError("回放消融要求全量预测")
    report = {"scope": "冻结单轮轨迹的模块关闭回放，非新检索/新模型实验；时延沿用原轨迹，非消融时延实测",
              "partial": len(records) != len(labels), "variants": {},
              "fingerprints": {"predictions": file_fingerprint(args.predictions),
                               "labels_source": file_fingerprint(args.labels_source)}}
    for variant in WorkflowAblations.VARIANTS:
        rows = WorkflowAblations.replay(records, variant)
        report["variants"][variant] = WorkflowScorer.score_records(rows, labels)
    ArtifactStore.atomic_text(Path(args.output), json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v["paired_hit"] for k, v in report["variants"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
