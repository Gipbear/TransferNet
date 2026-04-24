"""Run the simple WebQSP QA agent on one question."""

from __future__ import annotations

import argparse
import json

from oh_my_agent.agent import SimpleWebQAgent
from oh_my_agent.tools import AnswerWithPathsTool, PathRetrievalTool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the simple WebQSP QA agent on one sample")
    parser.add_argument("--question", required=True, help="Natural-language question")
    parser.add_argument("--topic_mid", required=True, help="Topic MID from WebQSP")
    parser.add_argument("--hop", type=int, default=None)
    parser.add_argument("--beam_size", type=int, default=20)
    parser.add_argument("--lambda_val", type=float, default=0.2)
    parser.add_argument("--prediction_threshold", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--path_server_url", default="http://localhost:8787")
    parser.add_argument("--llm_server_url", default="http://localhost:8788")
    parser.add_argument(
        "--entity_map",
        default="data/resources/WebQSP/fbwq_full/mapped_entities.txt",
        help="MID->name mapping file",
    )
    parser.add_argument("--no_adapter", action="store_true", help="Use the base model instead of the adapter")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path_tool = PathRetrievalTool(base_url=args.path_server_url, entity_map_path=args.entity_map)
    answer_tool = AnswerWithPathsTool(
        base_url=args.llm_server_url,
        default_use_adapter=not args.no_adapter,
        default_max_new_tokens=args.max_new_tokens,
    )
    agent = SimpleWebQAgent(path_tool=path_tool, answer_tool=answer_tool)
    result = agent.run(
        args.question,
        args.topic_mid,
        hop=args.hop,
        beam_size=args.beam_size,
        lambda_val=args.lambda_val,
        prediction_threshold=args.prediction_threshold,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
