"""Run the checked-batch WebQSP QA agent on one question."""

from __future__ import annotations

import argparse
import json

from oh_my_agent.agent import CheckedBatchWebQAgent
from oh_my_agent.tools import AnswerWithPathsTool, CitedPathCheckTool, PathRetrieveTool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the checked-batch WebQSP QA agent")
    parser.add_argument("--question", required=True, help="Natural-language question")
    parser.add_argument("--topic_mid", required=True, help="Topic MID from WebQSP")
    parser.add_argument("--path_method", choices=["tail_blend", "baseline"], default="tail_blend")
    parser.add_argument("--alpha_final", type=float, default=1.0)
    parser.add_argument("--path_threshold", type=float, default=0.01)
    parser.add_argument("--beam_size", type=int, default=50)
    parser.add_argument("--lambda_val", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--check_max_new_tokens", type=int, default=2)
    parser.add_argument("--path_retrieve_url", default="http://localhost:8789")
    parser.add_argument("--llm_server_url", default="http://localhost:8788")
    parser.add_argument(
        "--entity_map",
        default="data/resources/WebQSP/fbwq_full/mapped_entities.txt",
        help="MID->name mapping file",
    )
    parser.add_argument("--no_adapter", action="store_true", help="Use the base model for answering")
    parser.add_argument("--check_use_adapter", action="store_true", help="Use the adapter for path checks")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path_tool = PathRetrieveTool(
        base_url=args.path_retrieve_url,
        entity_map_path=args.entity_map,
    )
    answer_tool = AnswerWithPathsTool(
        base_url=args.llm_server_url,
        default_use_adapter=not args.no_adapter,
        default_max_new_tokens=args.max_new_tokens,
    )
    check_tool = CitedPathCheckTool(
        base_url=args.llm_server_url,
        default_use_adapter=args.check_use_adapter,
        default_max_new_tokens=args.check_max_new_tokens,
    )
    agent = CheckedBatchWebQAgent(
        path_tool=path_tool,
        answer_tool=answer_tool,
        check_tool=check_tool,
    )
    result = agent.run(
        args.question,
        args.topic_mid,
        method=args.path_method,
        alpha_final=args.alpha_final,
        threshold=args.path_threshold,
        beam_size=args.beam_size,
        lambda_val=args.lambda_val,
        batch_size=args.batch_size,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
