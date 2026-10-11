"""生成无标签 CWQ 运行输入和只依赖训练集的冻结模板包。"""

import argparse
import json
from pathlib import Path

from kgqa.agent.workflow.preparation import InputPreparation
from kgqa.agent.workflow.templates import TemplateLibrary
from kgqa.runtime import file_fingerprint


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="准备 CWQ 三路工作流输入与训练模板")
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--rog_dir", default="data/input/RoG-CWQ")
    parser.add_argument("--mid_dir", default="data/input/CWQ")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["test", "dev"], default="test")
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = InputPreparation(args.rog_dir, args.mid_dir).prepare(
        args.retrieval, output / "runtime_inputs_no_gold.jsonl", args.split
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)
    library = TemplateLibrary.fit(args.train)
    library.save(output / "templates.joblib")
    report.update({"training_n": library.training_n, "numeric_chain_n": len(library.numeric_index.labels),
                   "train": file_fingerprint(args.train), "retrieval": file_fingerprint(args.retrieval),
                   "runtime_input": file_fingerprint(output / "runtime_inputs_no_gold.jsonl"),
                   "templates": file_fingerprint(output / "templates.joblib")})
    (output / "preparation_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
