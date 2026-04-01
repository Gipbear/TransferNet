"""
Unsloth QLoRA 微调脚本（第四章 SFT 训练）

模型：unsloth/meta-llama-3.1-8b-instruct-bnb-4bit（预量化 4-bit）
方法：QLoRA（4-bit 量化 + LoRA）+ Prompt Masking（只在 assistant 回复部分计算 loss）

支持格式：V1（仅答案）、V2（路径引用，主方法）、V3（JSON）、V4（CoT）
输入数据：build_kgcot_dataset.py 生成的 JSONL（messages 字段，chat 格式）

用法示例：
  # WebQSP V2 主方法
  python llm_infer/train_sft.py \\
      --train data/output/WebQSP/kgcot_v2_train.jsonl \\
      --output_dir models/webqsp_v2 \\
      --epochs 3

  # MetaQA 采样子集
  python llm_infer/train_sft.py \\
      --train data/output/MetaQA/kgcot_v2_20k.jsonl \\
      --output_dir models/metaqa_v2_20k

  # 指定自定义超参
  python llm_infer/train_sft.py \\
      --train data/output/CWQ/kgcot_v2_train.jsonl \\
      --output_dir models/cwq_v2 \\
      --lr 1e-4 --batch_size 2 --grad_accum 16 --epochs 5
"""

import argparse
import json
import logging
import os
# 必须在所有 transformers/unsloth 导入之前设置
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['UNSLOTH_DISABLE_STATS'] = '1'
import sys
from datetime import datetime


# ─── 日志 ─────────────────────────────────────────────────────────────────────

def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("train_sft")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ─── 参数 ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Unsloth QLoRA SFT 微调")
    p.add_argument("--train",      required=True,  help="训练集 JSONL（messages 格式）")
    p.add_argument("--output_dir", required=True,  help="模型输出目录")
    p.add_argument("--model",      default="unsloth/meta-llama-3.1-8b-instruct-bnb-4bit")
    # LoRA 超参
    p.add_argument("--lora_rank",  type=int,   default=16)
    p.add_argument("--lora_alpha", type=int,   default=32)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    # 训练超参
    p.add_argument("--lr",         type=float, default=2e-4)
    p.add_argument("--batch_size", type=int,   default=4)
    p.add_argument("--grad_accum", type=int,   default=8,
                   help="梯度累积步数（有效 batch = batch_size * grad_accum）")
    p.add_argument("--epochs",     type=int,   default=5)
    p.add_argument("--max_seq_len", type=int,  default=1024+256)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


# ─── 数据加载 ──────────────────────────────────────────────────────────────────

def load_dataset_from_jsonl(path: str, tokenizer, max_seq_len: int, log: logging.Logger):
    """
    加载 JSONL 并转换为 HuggingFace Dataset。
    使用 apply_chat_template 序列化 messages，再做 Prompt Masking：
    只在 assistant 回复部分计算 loss（输入部分 label=-100）。

    截断策略（优先保留 golden paths）：
      1. 若完整 prompt 在预算内，直接使用。
      2. 若超长，通过 _meta.golden_path_indices 识别 distractor 路径行并逐条丢弃，
         直到 prompt 满足预算，确保 golden paths 被保留。
      3. 若移除所有 distractor 后仍超长，做 token 级截断兜底。
    """
    import re as _re
    import random
    from datasets import Dataset

    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    log.info("加载 %d 条训练样本", len(records))

    _PATH_LINE_RE = _re.compile(r"^(\d+)[\s:\[]")
    truncated_count = [0]

    def _drop_distractors_until_fits(messages: list, golden_indices: set,
                                     budget: int):
        """从 user 消息末尾逐条移除 distractor 路径行，直到 prompt_ids 满足 budget。"""
        user_text = messages[1]["content"]
        lines     = user_text.split("\n")

        path_line_positions = []
        for pos, line in enumerate(lines):
            m = _PATH_LINE_RE.match(line)
            if m:
                path_line_positions.append((pos, int(m.group(1))))

        distractor_positions = [
            pos for pos, num in path_line_positions
            if num not in golden_indices
        ]

        removed = set()
        for pos in reversed(distractor_positions):
            removed.add(pos)
            new_user = "\n".join(l for i, l in enumerate(lines) if i not in removed)
            new_msgs = [messages[0], {"role": "user", "content": new_user}]
            pt = tokenizer.apply_chat_template(
                new_msgs, tokenize=False, add_generation_prompt=True
            )
            pids = tokenizer(pt, add_special_tokens=False)["input_ids"]
            if len(pids) <= budget:
                return pids

        # 全部 distractor 移除后仍超长，返回当前最短版本（外层做 token 截断）
        new_user = "\n".join(l for i, l in enumerate(lines) if i not in removed)
        new_msgs = [messages[0], {"role": "user", "content": new_user}]
        pt = tokenizer.apply_chat_template(
            new_msgs, tokenize=False, add_generation_prompt=True
        )
        return tokenizer(pt, add_special_tokens=False)["input_ids"]

    def tokenize(rec):
        messages       = rec["messages"]
        meta           = rec.get("_meta", {})
        golden_indices = set(meta.get("golden_path_indices", []))

        asst_text    = messages[-1]["content"]
        asst_ids     = tokenizer(asst_text, add_special_tokens=False)["input_ids"]
        asst_reserve = len(asst_ids) + 8
        prompt_budget = max_seq_len - asst_reserve

        prompt_text = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        if len(prompt_ids) > prompt_budget:
            truncated_count[0] += 1
            # 路径级截断：逐条丢弃 distractor，优先保留 golden paths
            prompt_ids = _drop_distractors_until_fits(
                messages[:-1], golden_indices, prompt_budget
            )
            if len(prompt_ids) > prompt_budget:
                # 兜底 token 截断
                prompt_ids = prompt_ids[:prompt_budget]

        eot_id   = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        end_id   = eot_id if eot_id != tokenizer.unk_token_id else tokenizer.eos_token_id
        full_ids = prompt_ids + asst_ids + [end_id]
        full_ids = full_ids[:max_seq_len]

        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_len + full_ids[prompt_len:]

        if all(l == -100 for l in labels):
            keep   = min(asst_reserve, len(full_ids))
            labels = [-100] * (len(full_ids) - keep) + full_ids[-keep:]

        if random.random() < 0.005:
            valid = sum(1 for l in labels if l != -100)
            total = len(full_ids)
            print(f"[DEBUG] seq_len={total}, valid_labels={valid}, "
                  f"ratio={valid/total:.2%}, prompt_len={prompt_len}")

        return {
            "input_ids":      full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels":         labels,
        }

    raw = Dataset.from_list([
        {"messages": r["messages"], "_meta": r.get("_meta", {})}
        for r in records
    ])
    tokenized = raw.map(tokenize, remove_columns=["messages", "_meta"])
    log.info("截断样本数: %d / %d (%.1f%%)",
             truncated_count[0], len(records),
             100 * truncated_count[0] / len(records) if records else 0)
    return tokenized


# ─── 主训练流程 ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.output_dir, f"train_{run_id}.log")
    log      = setup_logger(log_path)
    log.info("命令: %s", " ".join(sys.argv))
    log.info("模型: %s", args.model)
    log.info("训练集: %s", args.train)
    log.info("输出目录: %s", args.output_dir)
    log.info("LoRA rank=%d alpha=%d  lr=%.2e  batch=%d  grad_accum=%d  epochs=%d",
             args.lora_rank, args.lora_alpha, args.lr,
             args.batch_size, args.grad_accum, args.epochs)

    # ── 加载 Unsloth 模型 ────────────────────────────────────────────────────
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        sys.exit("[Error] unsloth 未安装。请运行: pip install unsloth")

    import torch

    # 优先使用本地缓存，避免无网络环境触发 API 请求
    model_path = args.model

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=args.max_seq_len,
        dtype=None,
        load_in_4bit=True,
        local_files_only=True,
    )

    # ── 注入 LoRA ────────────────────────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    log.info("LoRA 注入完成")

    # ── 加载数据集（含 Prompt Masking）──────────────────────────────────────
    train_dataset = load_dataset_from_jsonl(
        args.train, tokenizer, args.max_seq_len, log
    )

    # ── 训练配置 ─────────────────────────────────────────────────────────────
    from transformers import TrainingArguments
    from trl import SFTTrainer

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="no",
        seed=args.seed,
        report_to="none",
        dataloader_num_workers=0,
    )

    # SFTTrainer：直接传入已经 tokenize（含 labels）的数据集
    # dataset_kwargs skip_prepare_dataset=True 确保 SFTTrainer 不重新处理 labels
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        dataset_text_field=None,   # 已手动 tokenize，跳过内部处理
        max_seq_length=args.max_seq_len,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    log.info("开始训练...")
    trainer_stats = trainer.train()
    log.info("训练完成: %s", trainer_stats)

    # ── 保存模型 ─────────────────────────────────────────────────────────────
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    log.info("模型已保存至: %s", args.output_dir)
    log.info("日志: %s", log_path)


if __name__ == "__main__":
    main()
