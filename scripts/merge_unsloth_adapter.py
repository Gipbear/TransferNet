"""Merge an Unsloth/PEFT LoRA adapter into a vLLM-friendly model directory."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model",
        default="unsloth/meta-llama-3.1-8b-instruct-bnb-4bit",
        help="Base model name or local snapshot path.",
    )
    parser.add_argument(
        "--adapter",
        default="models/webqsp/ablation/groupJ_schema_name",
        help="PEFT/Unsloth LoRA adapter directory.",
    )
    parser.add_argument(
        "--output",
        default="models/webqsp/ablation/groupJ_schema_name_merged_16bit",
        help="Output directory for the merged model.",
    )
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument(
        "--save-method",
        choices=["merged_16bit", "merged_4bit", "lora"],
        default="merged_16bit",
    )
    parser.add_argument(
        "--maximum-memory-usage",
        type=float,
        default=0.5,
        help="Passed to Unsloth save_pretrained_merged when supported.",
    )
    parser.add_argument(
        "--load-adapter-as-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the adapter directory directly with FastLanguageModel.from_pretrained.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    from unsloth import FastLanguageModel

    model_name = args.adapter if args.load_adapter_as_model else args.base_model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
        local_files_only=True,
    )

    if not args.load_adapter_as_model:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs = {"save_method": args.save_method}
    if args.maximum_memory_usage is not None:
        save_kwargs["maximum_memory_usage"] = args.maximum_memory_usage

    try:
        model.save_pretrained_merged(str(output), tokenizer, **save_kwargs)
    except TypeError:
        save_kwargs.pop("maximum_memory_usage", None)
        model.save_pretrained_merged(str(output), tokenizer, **save_kwargs)

    print(f"saved merged model: {output}")


if __name__ == "__main__":
    main()
