#!/usr/bin/env python3
"""Build a WebQSP QA file that only keeps rows with valid answer entities."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_INPUT_DIR = Path("data/input/WebQSP")
DEFAULT_SOURCE = DEFAULT_INPUT_DIR / "QA_data/WebQuestionsSP/qa_test_webqsp_fixed.txt"
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "QA_data/WebQuestionsSP/qa_test_webqsp_fixed_1581.txt"
DEFAULT_ENTITIES = DEFAULT_INPUT_DIR / "fbwq_full/entities.dict"


def load_entities(path: Path) -> set[str]:
    entities: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entities.add(line.split("\t", 1)[0].strip())
    return entities


def row_has_valid_answer(line: str, entities: set[str]) -> bool:
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 2:
        return False
    answers = [answer.strip() for answer in parts[1].split("|") if answer.strip()]
    return any(answer in entities for answer in answers)


def build_fixed_qa(source: Path, output: Path, entities_path: Path, force: bool) -> tuple[int, int]:
    if output.exists() and not force:
        raise FileExistsError(f"output exists, pass --force to overwrite: {output}")

    entities = load_entities(entities_path)
    total = 0
    kept = 0
    output.parent.mkdir(parents=True, exist_ok=True)

    with source.open(encoding="utf-8") as src, output.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            total += 1
            if row_has_valid_answer(line, entities):
                dst.write(line)
                kept += 1

    return total, kept


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter WebQSP QA rows to those with at least one answer in entities.dict."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--entities", type=Path, default=DEFAULT_ENTITIES)
    parser.add_argument("--force", action="store_true", help="overwrite output if it already exists")
    args = parser.parse_args()

    total, kept = build_fixed_qa(args.source, args.output, args.entities, args.force)
    print(f"[INFO] wrote {kept}/{total} rows to {args.output}")


if __name__ == "__main__":
    main()
