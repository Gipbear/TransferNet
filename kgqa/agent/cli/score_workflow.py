"""独立读取金答案，对已落盘工作流输出做配对评分。"""

import argparse
import json
from pathlib import Path

from kgqa.agent.workflow.scoring import WorkflowScorer
from kgqa.runtime import file_fingerprint


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CWQ 工作流独立评分")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--labels_source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow_partial", action="store_true")
    args = parser.parse_args(argv)
    report = WorkflowScorer.score_file(Path(args.predictions), Path(args.labels_source), args.allow_partial)
    report["inputs"] = {"predictions": file_fingerprint(args.predictions),
                        "labels": file_fingerprint(args.labels_source)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("overall", "baseline", "paired_hit", "partial")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
