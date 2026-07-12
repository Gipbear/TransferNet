"""pfit QLoRA 训练:sft_train.jsonl → LoRA adapter(迁自 llm_infer/train_sft.py)。

模型 unsloth/meta-llama-3.1-8b-instruct-bnb-4bit,QLoRA + Prompt Masking
(只在 assistant 回复部分计算 loss)。数据整形函数为模块级纯函数,免 GPU 可测;
Unsloth/训练主体依赖 GPU,由 smoke 验证。

用法:
  python -m kgqa.pfit.train --exp_dir data/output/kgqa/webqsp/pfit/webqsp_main --epochs 2
"""
from __future__ import annotations

import argparse
import json
import logging
import os
# 必须在所有 transformers/unsloth 导入之前设置
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("UNSLOTH_DISABLE_STATS", "1")
import random
import re
import sys
from datetime import datetime

from kgqa.pfit import manifest as manifest_mod

log = logging.getLogger("pfit.train")

_PATH_LINE_RE = re.compile(r"^(\d+)[\s:\[]")


# ─── 数据整形(纯函数,免 GPU) ──────────────────────────────────────────────────

def drop_distractors_until_fits(messages: list, golden_indices: set,
                                budget: int, tokenizer) -> list:
    """从 user 消息中逐条移除 distractor 路径行,直到 prompt 满足预算。

    golden 路径行不移除;全部 distractor 移除后仍超长时返回当前版本,
    由调用方做 token 级截断兜底。
    """
    user_text = messages[1]["content"]
    lines = user_text.split("\n")

    path_line_positions = []
    for pos, line in enumerate(lines):
        m = _PATH_LINE_RE.match(line)
        if m:
            path_line_positions.append((pos, int(m.group(1))))

    distractor_positions = [
        pos for pos, num in path_line_positions
        if num not in golden_indices
    ]
    random.shuffle(distractor_positions)

    removed = set()
    for pos in distractor_positions:
        removed.add(pos)
        new_user = "\n".join(l for i, l in enumerate(lines) if i not in removed)
        new_msgs = [messages[0], {"role": "user", "content": new_user}]
        pt = tokenizer.apply_chat_template(
            new_msgs, tokenize=False, add_generation_prompt=True
        )
        pids = tokenizer(pt, add_special_tokens=False)["input_ids"]
        if len(pids) <= budget:
            return pids

    new_user = "\n".join(l for i, l in enumerate(lines) if i not in removed)
    new_msgs = [messages[0], {"role": "user", "content": new_user}]
    pt = tokenizer.apply_chat_template(
        new_msgs, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(pt, add_special_tokens=False)["input_ids"]


def tokenize_record(rec: dict, tokenizer, max_seq_len: int) -> dict:
    """单样本 tokenize + Prompt Masking。

    截断策略(优先保留 golden paths):
      1. 完整 prompt 在预算内直接用;
      2. 超长时按 _meta.golden_path_indices 丢 distractor 行(truncated=True);
      3. 仍超长做 token 级截断,全 -100 时强制保留末尾 assistant 段 label
         (label_fallback=True)。
    """
    messages = rec["messages"]
    meta = rec.get("_meta", {})
    golden_indices = set(meta.get("golden_path_indices", []))

    asst_text = messages[-1]["content"]
    asst_ids = tokenizer(asst_text, add_special_tokens=False)["input_ids"]
    asst_reserve = len(asst_ids) + 1
    prompt_budget = max_seq_len - asst_reserve

    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    truncated = False
    if len(prompt_ids) > prompt_budget:
        truncated = True
        prompt_budget = max(0, prompt_budget)
        prompt_ids = drop_distractors_until_fits(
            messages[:-1], golden_indices, prompt_budget, tokenizer
        )
        if len(prompt_ids) > prompt_budget:
            prompt_ids = prompt_ids[:prompt_budget]

    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    end_id = eot_id if eot_id != tokenizer.unk_token_id else tokenizer.eos_token_id
    full_ids = prompt_ids + asst_ids + [end_id]
    full_ids = full_ids[:max_seq_len]

    prompt_len = min(len(prompt_ids), len(full_ids))
    labels = [-100] * prompt_len + full_ids[prompt_len:]

    label_fallback = False
    if all(l == -100 for l in labels):
        # prompt 占满 budget,强制保留末尾 asst_reserve 个 token 的 label
        label_fallback = True
        keep = min(asst_reserve, len(full_ids))
        labels = [-100] * (len(full_ids) - keep) + full_ids[-keep:]

    return {
        "input_ids":      full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels":         labels,
        "length":         len(full_ids),
        "truncated":      truncated,
        "label_fallback": label_fallback,
    }


def load_dataset_from_jsonl(path: str, tokenizer, max_seq_len: int):
    """JSONL → HuggingFace Dataset(tokenize + masking + 截断计数)。"""
    from datasets import Dataset

    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    log.info("加载 %d 条训练样本", len(records))

    counters = {"truncated": 0, "label_fallback": 0}

    def _tokenize(rec):
        out = tokenize_record(rec, tokenizer, max_seq_len)
        counters["truncated"] += out.pop("truncated")
        counters["label_fallback"] += out.pop("label_fallback")
        return out

    raw = Dataset.from_list([
        {"messages": r["messages"], "_meta": r.get("_meta", {})}
        for r in records
    ])
    tokenized = raw.map(_tokenize, remove_columns=["messages", "_meta"])

    n = len(records)
    tc, lf = counters["truncated"], counters["label_fallback"]
    log.info("截断样本数: %d / %d (%.1f%%)", tc, n, 100 * tc / n if n else 0)
    fallback_pct = 100 * lf / n if n else 0
    log.info("标签退化(全 -100 兜底)样本数: %d / %d (%.1f%%)%s",
             lf, n, fallback_pct,
             "  ⚠ 超过 5%,建议增大 --max_seq_len 或减少 distractor" if fallback_pct > 5 else "")
    return tokenized


# ─── 训练指标回调 ──────────────────────────────────────────────────────────────

def _jsonable_metric(value):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def make_metrics_callback(metrics_path: str):
    from transformers import TrainerCallback

    class MetricsJsonlCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or not getattr(state, "is_local_process_zero", True):
                return
            record = {
                "time": datetime.now().isoformat(timespec="seconds"),
                "step": getattr(state, "global_step", 0),
            }
            epoch = getattr(state, "epoch", None)
            if epoch is not None:
                record["epoch"] = _jsonable_metric(epoch)
            for key, value in logs.items():
                if key != "total_flos":
                    record[key] = _jsonable_metric(value)
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    os.makedirs(os.path.dirname(os.path.abspath(metrics_path)), exist_ok=True)
    return MetricsJsonlCallback()


# ─── 主训练流程(GPU) ──────────────────────────────────────────────────────────

def run_train(*, exp_dir: str, train_file: str = None,
              model: str = "unsloth/meta-llama-3.1-8b-instruct-bnb-4bit",
              lora_rank: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.0,
              lr: float = 2e-4, batch_size: int = 4, grad_accum: int = 8,
              epochs: int = 2, max_seq_len: int = 1024 + 256,
              warmup_ratio: float = 0.05, seed: int = 42,
              val_ratio: float = 0.05) -> str:
    """训练主入口,adapter 写 exp_dir/adapter/;同配置且 adapter 已在则跳过。"""
    train_file = train_file or os.path.join(exp_dir, "sft_train.jsonl")
    adapter_dir = os.path.join(exp_dir, "adapter")
    manifest_path = os.path.join(exp_dir, "manifest.json")

    config = {
        "model": model, "lora_rank": lora_rank, "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout, "lr": lr, "batch_size": batch_size,
        "grad_accum": grad_accum, "epochs": epochs, "max_seq_len": max_seq_len,
        "warmup_ratio": warmup_ratio, "seed": seed, "val_ratio": val_ratio,
    }
    section = manifest_mod.make_section(config, {"sft_train": train_file})

    existing = manifest_mod.load(manifest_path).get("train")
    adapter_ready = os.path.isfile(os.path.join(adapter_dir, "adapter_config.json"))
    if existing is not None:
        if manifest_mod.sections_compatible(existing, section) and adapter_ready:
            log.info("train 已完成且配置一致,跳过:%s", adapter_dir)
            return adapter_dir
        if not manifest_mod.sections_compatible(existing, section):
            raise RuntimeError(
                f"{exp_dir} 已有不同配置的 train 记录;请换 exp_dir 或删除旧目录后重跑")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = os.path.join(exp_dir, f"train_metrics_{run_id}.jsonl")
    log.info("训练集: %s  adapter 输出: %s", train_file, adapter_dir)
    log.info("LoRA rank=%d alpha=%d  lr=%.2e  batch=%d  grad_accum=%d  epochs=%d",
             lora_rank, lora_alpha, lr, batch_size, grad_accum, epochs)

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        sys.exit("[Error] unsloth 未安装。请运行: pip install unsloth")

    import torch
    from trl import SFTConfig, SFTTrainer

    model_obj, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model,
        max_seq_length=max_seq_len,
        dtype=None,
        load_in_4bit=True,
        local_files_only=True,
    )
    model_obj = FastLanguageModel.get_peft_model(
        model_obj,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )
    log.info("LoRA 注入完成")

    full_dataset = load_dataset_from_jsonl(train_file, tokenizer, max_seq_len)

    eval_dataset = None
    if val_ratio > 0:
        split = full_dataset.train_test_split(test_size=val_ratio, seed=seed)
        train_dataset, eval_dataset = split["train"], split["test"]
        log.info("验证集划分: train=%d  val=%d (val_ratio=%.2f)",
                 len(train_dataset), len(eval_dataset), val_ratio)
    else:
        train_dataset = full_dataset

    # 按序列长度升序排列,使同批次内长度相近,减少 padding 浪费
    train_dataset = train_dataset.sort("length")

    training_args = SFTConfig(
        output_dir=adapter_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="no",
        eval_strategy="epoch" if eval_dataset is not None else "no",
        per_device_eval_batch_size=batch_size,
        seed=seed,
        report_to="none",
        dataloader_num_workers=0,
        max_length=max_seq_len,
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    trainer = SFTTrainer(
        model=model_obj,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )
    trainer.add_callback(make_metrics_callback(metrics_path))

    log.info("开始训练...指标写入 %s", metrics_path)
    trainer_stats = trainer.train()
    log.info("训练完成: %s", trainer_stats)

    model_obj.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    manifest_mod.merge_section(manifest_path, "train", section)
    log.info("adapter 已保存: %s", adapter_dir)
    return adapter_dir


# ─── CLI ──────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(description="pfit QLoRA SFT 训练")
    p.add_argument("--exp_dir", required=True, help="实验目录(读 sft_train.jsonl,写 adapter/)")
    p.add_argument("--train_file", default=None, help="覆盖默认训练集路径")
    p.add_argument("--model", default="unsloth/meta-llama-3.1-8b-instruct-bnb-4bit")
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max_seq_len", type=int, default=1024 + 256)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_ratio", type=float, default=0.05)
    return p


def main(argv=None):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    a = build_parser().parse_args(argv)
    run_train(exp_dir=a.exp_dir, train_file=a.train_file, model=a.model,
              lora_rank=a.lora_rank, lora_alpha=a.lora_alpha,
              lora_dropout=a.lora_dropout, lr=a.lr, batch_size=a.batch_size,
              grad_accum=a.grad_accum, epochs=a.epochs, max_seq_len=a.max_seq_len,
              warmup_ratio=a.warmup_ratio, seed=a.seed, val_ratio=a.val_ratio)


if __name__ == "__main__":
    main()
