"""
忠实度评估脚本（第四章核心评估，支持 V0~V4 输出格式）

评估指标：
  答案准确率（与第三章对齐）：
    Hit@1        首个预测答案命中率
    F1           集合级别 Precision / Recall / F1
    Exact Match  预测答案集合与真实答案完全匹配

  忠实度指标（第四章新增）：
    Citation Accuracy   被引用路径中命中 Golden Path 的占比
    Citation Recall     Golden Path 被模型引用的占比
    Hallucination Rate  输出实体不存在于任何输入路径的比例
    Format Compliance   输出能被正确解析的比例

用法示例：
  # V2 主方法（微调模型 + LoRA adapter）
  python llm_infer/eval_faithfulness.py \\
      --input  data/output/WebQSP/predict_test.jsonl \\
      --output data/output/WebQSP/eval_v2 \\
      --adapter models/webqsp_v2 \\
      --output_format v2

  # V0 零样本基线（无 adapter）
  python llm_infer/eval_faithfulness.py \\
      --input  data/output/WebQSP/predict_test.jsonl \\
      --output data/output/WebQSP/eval_v0 \\
      --output_format v0

  # 抗噪鲁棒性实验：推理时增加额外干扰路径
  python llm_infer/eval_faithfulness.py \\
      --input  data/output/WebQSP/predict_test.jsonl \\
      --output data/output/WebQSP/eval_v2_noise15 \\
      --adapter models/webqsp_v2 \\
      --output_format v2 \\
      --noise_paths 15
"""

import argparse
import json
import logging
import os
# 必须在所有 transformers/unsloth 导入之前设置
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['UNSLOTH_DISABLE_STATS'] = '1'
import re
import sys
import warnings
from collections import defaultdict
from datetime import datetime

import torch
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)


# ─── Prompt ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a KGQA assistant. "
    "Given reasoning paths from a knowledge graph and a question, "
    "identify the supporting paths and output the answer."
)

# V0/V1 格式提示词（只输出答案）
SYSTEM_PROMPT_ANSWER_ONLY = (
    "You are a KGQA assistant. "
    "Given reasoning paths from a knowledge graph and a question, "
    "answer the question using entity IDs from the paths.\n"
    "Rules:\n"
    "- Only output entity IDs that appear in the provided paths.\n"
    "- Do not generate or fabricate new entity IDs.\n"
    "Output format:\nAnswer: <entity_id> | <entity_id>"
)

# V2 格式提示词（路径引用 + 答案）
SYSTEM_PROMPT_V2 = (
    "You are a KGQA assistant. "
    "Given reasoning paths from a knowledge graph and a question, "
    "identify which paths support the answer, then extract the answer "
    "from the tail entities of those supporting paths.\n"
    "Rules:\n"
    "- Only output entity IDs that appear in the provided paths.\n"
    "- Do not generate or fabricate new entity IDs.\n"
    "Output format:\n"
    "Supporting Paths: <path numbers>\n"
    "Answer: <entity_id> | <entity_id>"
)

# V3 格式提示词（JSON）
SYSTEM_PROMPT_V3 = (
    "You are a KGQA assistant. "
    "Given reasoning paths from a knowledge graph and a question, "
    "output a JSON object with the supporting path indices and the answer entity IDs.\n"
    "Rules:\n"
    "- Only output entity IDs that appear in the provided paths.\n"
    "- Do not generate or fabricate new entity IDs.\n"
    'Output format: {"reasoning": ["Path 1", "Path 3"], "answer": ["<entity_id>", "<entity_id>"]}'
)

# V4 格式提示词（CoT）
SYSTEM_PROMPT_V4 = (
    "You are a KGQA assistant. "
    "Given reasoning paths from a knowledge graph and a question, "
    "reason step by step about which paths support the answer, then output the answer entity IDs.\n"
    "Rules:\n"
    "- Only output entity IDs that appear in the provided paths.\n"
    "- Do not generate or fabricate new entity IDs.\n"
    "Output format:\n[Reasoning]\n...\n[Answer]\nSupporting Paths: 1, 3\nAnswer: <entity_id>"
)

FORMAT_PROMPTS = {
    "v0": SYSTEM_PROMPT_ANSWER_ONLY,
    "v1": SYSTEM_PROMPT_ANSWER_ONLY,
    "v2": SYSTEM_PROMPT_V2,
    "v3": SYSTEM_PROMPT_V3,
    "v4": SYSTEM_PROMPT_V4,
}


# ─── 日志 ─────────────────────────────────────────────────────────────────────

def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("eval_faithfulness")
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


# ─── 路径格式化（与 build_kgcot_dataset.py 保持一致）────────────────────────

def format_path_str(path_edges: list, log_score: float, idx: int) -> str:
    chain = " ".join(f"({e[0]}) -[{e[1]}]-> ({e[2]})" for e in path_edges)
    return f"Path {idx} [score={log_score:.4f}]: {chain}"


def build_user_content(paths_with_meta: list, question: str) -> str:
    lines = [f"Question: {question}", "", "Reasoning Paths:"]
    for path_edges, log_score, display_idx in paths_with_meta:
        lines.append(format_path_str(path_edges, log_score, display_idx))
    return "\n".join(lines)


# ─── Golden Path 标注（推理时使用，用于计算忠实度指标）───────────────────────

def label_golden_indices(mmr_paths: list, golden: list) -> set:
    """返回 1-based display index 集合：尾实体在 golden 中的路径。"""
    golden_set = {g.lower().strip() for g in golden}
    result = set()
    for i, p in enumerate(mmr_paths):
        edges = p.get("path", [])
        tail  = edges[-1][2].lower().strip() if edges else None
        if tail and tail in golden_set:
            result.add(i + 1)
    return result


def get_all_path_entities(mmr_paths: list) -> set:
    """收集所有路径中出现过的实体（用于幻觉检测）。"""
    entities = set()
    for p in mmr_paths:
        for edge in p.get("path", []):
            entities.add(edge[0].lower().strip())
            entities.add(edge[2].lower().strip())
    return entities


# ─── 输出解析 ──────────────────────────────────────────────────────────────────

_ANSWER_RE     = re.compile(r"Answer\s*[:：]\s*(.+)", re.IGNORECASE)
_CITE_RE       = re.compile(r"Supporting\s*Paths?\s*[:：]\s*([\d,\s]+)", re.IGNORECASE)
_JSON_RE       = re.compile(r"\{.*\}", re.DOTALL)


def parse_output(raw: str, fmt: str) -> dict:
    """
    解析模型输出，返回：
      answers:        预测答案实体列表
      cited_indices:  被引用的路径编号集合（1-based）
      format_ok:      输出格式是否合规
    """
    raw = raw.strip()

    def _dedup(lst: list) -> list:
        """保序去重（dict.fromkeys 保留首次出现顺序）。"""
        return list(dict.fromkeys(lst))

    _PLACEHOLDER_RE = re.compile(r"^entity\d*$", re.IGNORECASE)

    def _parse_answers(ans_raw: str) -> list:
        return _dedup(
            e.strip().strip('"\'[]') for e in re.split(r"[|,]", ans_raw)
            if e.strip() and not _PLACEHOLDER_RE.match(e.strip().strip('"\'[]'))
        )

    if fmt in ("v0", "v1"):
        m = _ANSWER_RE.search(raw)
        if m:
            ans_raw = m.group(1).strip().splitlines()[0]
            return {"answers": _parse_answers(ans_raw), "cited_indices": set(), "format_ok": True}
        # fallback
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        answers = _dedup(e.strip() for e in re.split(r"[|,]", lines[-1]) if e.strip()) if lines else []
        return {"answers": answers, "cited_indices": set(), "format_ok": False}

    elif fmt == "v2":
        # 期望: "Supporting Paths: 1, 3\nAnswer: entity"
        cite_m   = _CITE_RE.search(raw)
        answer_m = _ANSWER_RE.search(raw)
        format_ok = bool(cite_m and answer_m)

        cited_indices = set()
        if cite_m:
            for tok in re.split(r"[,\s]+", cite_m.group(1)):
                tok = tok.strip()
                if tok.isdigit():
                    cited_indices.add(int(tok))

        answers = _parse_answers(answer_m.group(1).strip().splitlines()[0]) if answer_m else []
        return {"answers": answers, "cited_indices": cited_indices, "format_ok": format_ok}

    elif fmt == "v3":
        # 期望: {"reasoning": ["Path 1"], "answer": ["entity"]}
        jm = _JSON_RE.search(raw)
        if jm:
            try:
                obj = json.loads(jm.group())
                answers = _dedup(obj.get("answer", []))
                reasoning_strs = obj.get("reasoning", [])
                cited_indices = set()
                for rs in reasoning_strs:
                    nums = re.findall(r"\d+", str(rs))
                    cited_indices.update(int(n) for n in nums)
                return {"answers": answers, "cited_indices": cited_indices, "format_ok": True}
            except json.JSONDecodeError:
                pass
        # fallback
        answer_m = _ANSWER_RE.search(raw)
        answers  = _parse_answers(answer_m.group(1).strip().splitlines()[0]) if answer_m else []
        return {"answers": answers, "cited_indices": set(), "format_ok": False}

    elif fmt == "v4":
        # 期望: [Reasoning]...[Answer]\nSupporting Paths: ...\nAnswer: ...
        cite_m   = _CITE_RE.search(raw)
        answer_m = _ANSWER_RE.search(raw)
        has_reasoning = "[Reasoning]" in raw or "[reasoning]" in raw.lower()
        format_ok = bool(cite_m and answer_m and has_reasoning)

        cited_indices = set()
        if cite_m:
            for tok in re.split(r"[,\s]+", cite_m.group(1)):
                tok = tok.strip()
                if tok.isdigit():
                    cited_indices.add(int(tok))

        answers = _parse_answers(answer_m.group(1).strip().splitlines()[0]) if answer_m else []
        return {"answers": answers, "cited_indices": cited_indices, "format_ok": format_ok}

    else:
        raise ValueError(f"未知格式: {fmt}")


# ─── 答案准确率指标 ────────────────────────────────────────────────────────────

def norm_entity(s: str) -> str:
    return s.lower().strip()


def compute_answer_metrics(pred: list, gold: list) -> dict:
    pred_set = {norm_entity(e) for e in pred if e.strip()}
    gold_set = {norm_entity(e) for e in gold if e.strip()}

    hit1     = int(bool(pred) and norm_entity(pred[0]) in gold_set)
    hit_any  = int(bool(pred_set & gold_set))

    if not pred_set and not gold_set:
        return {"hit1": 1, "hit_any": 1, "precision": 1.0, "recall": 1.0, "f1": 1.0,
                "exact_match": True, "tp": 0, "pred_n": 0, "gold_n": 0}
    if not pred_set or not gold_set:
        return {"hit1": hit1, "hit_any": hit_any, "precision": 0.0, "recall": 0.0,
                "f1": 0.0, "exact_match": False,
                "tp": 0, "pred_n": len(pred_set), "gold_n": len(gold_set)}

    tp = len(pred_set & gold_set)
    p  = tp / len(pred_set)
    r  = tp / len(gold_set)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"hit1": hit1, "hit_any": hit_any, "precision": p, "recall": r, "f1": f1,
            "exact_match": pred_set == gold_set,
            "tp": tp, "pred_n": len(pred_set), "gold_n": len(gold_set)}


# ─── 忠实度指标 ────────────────────────────────────────────────────────────────

def compute_faithfulness(cited_indices: set, golden_indices: set,
                         pred_answers: list, path_entities: set) -> dict:
    """
    citation_accuracy  被引用路径中命中 Golden 的占比
    citation_recall    Golden Path 被引用的占比
    hallucination      输出实体不存在于任何路径的比例
    """
    # Citation Accuracy
    if cited_indices:
        cit_acc = len(cited_indices & golden_indices) / len(cited_indices)
    else:
        cit_acc = 0.0  # 未引用任何路径

    # Citation Recall
    if golden_indices:
        cit_rec = len(cited_indices & golden_indices) / len(golden_indices)
    else:
        cit_rec = 0.0  # 无 Golden Path 的样本（path_hit=False），recall 定义为 0

    # Hallucination：预测实体不在路径中
    if pred_answers:
        hallu_entities = [e for e in pred_answers
                          if norm_entity(e) not in path_entities]
        hallu_rate = len(hallu_entities) / len(pred_answers)
    else:
        hallu_entities = []
        hallu_rate     = 0.0

    return {
        "citation_accuracy":   round(cit_acc, 4),
        "citation_recall":     round(cit_rec, 4),
        "hallucination_rate":  round(hallu_rate, 4),
        "hallucinated_entities": hallu_entities,
    }


# ─── 汇总 ─────────────────────────────────────────────────────────────────────

def aggregate(results: list) -> dict:
    n = len(results)
    if n == 0:
        return {}

    def mean(key):
        return round(sum(r[key] for r in results) / n, 4)

    macro_p   = mean("precision")
    macro_r   = mean("recall")
    macro_f1  = mean("f1")
    hit1      = mean("hit1")
    hit_any   = sum(1 for r in results if r["hit_any"]) / n
    exact     = sum(1 for r in results if r["exact_match"]) / n

    tp_sum   = sum(r["tp"]     for r in results)
    pred_sum = sum(r["pred_n"] for r in results)
    gold_sum = sum(r["gold_n"] for r in results)
    micro_p  = tp_sum / pred_sum if pred_sum > 0 else 0.0
    micro_r  = tp_sum / gold_sum if gold_sum > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    # 忠实度
    cit_acc  = mean("citation_accuracy")
    cit_rec  = mean("citation_recall")
    hallu    = mean("hallucination_rate")
    fmt_comp = sum(1 for r in results if r["format_ok"]) / n

    return {
        "n": n,
        "hit1":         round(hit1,    4),
        "hit_any":      round(hit_any, 4),
        "macro_p":      round(macro_p, 4),
        "macro_r":      round(macro_r, 4),
        "macro_f1":     round(macro_f1, 4),
        "micro_p":      round(micro_p,  4),
        "micro_r":      round(micro_r,  4),
        "micro_f1":     round(micro_f1, 4),
        "exact_match":  round(exact,    4),
        "citation_accuracy":  round(cit_acc, 4),
        "citation_recall":    round(cit_rec, 4),
        "hallucination_rate": round(hallu,   4),
        "format_compliance":  round(fmt_comp, 4),
    }


def log_metrics(log: logging.Logger, title: str, m: dict):
    log.info("  [ %s ]  (n=%d)", title, m["n"])
    log.info("    Hit@1             : %.4f", m["hit1"])
    log.info("    Hit@Any           : %.4f", m["hit_any"])
    log.info("    Macro  P/R/F1     : %.4f / %.4f / %.4f",
             m["macro_p"], m["macro_r"], m["macro_f1"])
    log.info("    Micro  P/R/F1     : %.4f / %.4f / %.4f",
             m["micro_p"], m["micro_r"], m["micro_f1"])
    log.info("    Exact Match       : %.4f", m["exact_match"])
    log.info("    Citation Accuracy : %.4f", m["citation_accuracy"])
    log.info("    Citation Recall   : %.4f", m["citation_recall"])
    log.info("    Hallucination Rate: %.4f", m["hallucination_rate"])
    log.info("    Format Compliance : %.4f", m["format_compliance"])


# ─── 主函数 ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="第四章忠实度评估")
    p.add_argument("--input",         required=True, help="predict JSONL（含 mmr_reason_paths / golden）")
    p.add_argument("--output",        default=None,  help="输出目录（文件名自动生成）")
    p.add_argument("--model",         default="unsloth/meta-llama-3.1-8b-instruct-bnb-4bit")
    p.add_argument("--adapter",       default=None,  help="LoRA adapter 目录（微调模型）")
    p.add_argument("--output_format", default="v2",
                   choices=["v0", "v1", "v2", "v3", "v4"],
                   help="模型输出格式（决定 prompt 和解析方式）")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--batch_size",     type=int, default=8,
                   help="批量推理大小（越大越快，受显存限制；建议 4~16）")
    p.add_argument("--limit",          type=int, default=0, help="只评估前 N 条（0=全部）")
    p.add_argument("--noise_paths",    type=int, default=0,
                   help="推理时在输入末尾追加 N 条伪干扰路径（抗噪鲁棒性实验）")
    return p.parse_args()


def resolve_output(input_path: str, output_arg, fmt: str, adapter: str) -> str:
    stem   = os.path.splitext(os.path.basename(input_path))[0]
    suffix = f"_{fmt}"
    if adapter:
        suffix += "_ft"
    suffix += "_eval.jsonl"
    outdir = output_arg if output_arg else os.path.dirname(os.path.abspath(input_path))
    return os.path.join(outdir, stem + suffix)


def main():
    args = parse_args()

    output_path = resolve_output(args.input, args.output, args.output_format, args.adapter)
    log_path    = os.path.splitext(output_path)[0] + ".log"
    log         = setup_logger(log_path)

    log.info("=" * 60)
    log.info("eval_faithfulness 启动")
    log.info("  命令          : %s", " ".join(sys.argv))
    log.info("  input         : %s", args.input)
    log.info("  output        : %s", output_path)
    log.info("  model         : %s", args.model)
    log.info("  adapter       : %s", args.adapter or "None（零样本）")
    log.info("  output_format : %s", args.output_format)
    log.info("  max_new_tokens: %d", args.max_new_tokens)
    log.info("  limit         : %s", args.limit if args.limit > 0 else "全部")
    log.info("  batch_size    : %d", args.batch_size)
    log.info("  noise_paths   : %d", args.noise_paths)
    log.info("=" * 60)

    # ── 加载模型 ─────────────────────────────────────────────────────────────
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        sys.exit("[Error] unsloth 未安装。请运行: pip install unsloth")

    model_path = args.model

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        local_files_only=True,
    )

    if args.adapter:
        log.info("加载 LoRA adapter: %s", args.adapter)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)

    FastLanguageModel.for_inference(model)
    model.eval()

    # ── 读取数据 ──────────────────────────────────────────────────────────────
    with open(args.input, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    if args.limit > 0:
        samples = samples[:args.limit]
    log.info("样本数: %d", len(samples))

    # 选择 system prompt
    system_prompt = FORMAT_PROMPTS[args.output_format]

    # 批量推理需要左填充（decoder-only 模型）
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 预处理：构建每条样本的 prompt 和 meta 信息 ────────────────────────────
    def prepare_sample(sample):
        question  = sample.get("question", "")
        mmr_paths = list(sample.get("mmr_reason_paths", []))
        golden    = sample.get("golden", [])

        if args.noise_paths > 0 and mmr_paths:
            import random
            existing = list(mmr_paths)
            for i in range(args.noise_paths):
                base = existing[i % len(existing)]
                fake = [[f"noise_{i}_{j}", e[1], f"noise_{i}_{j+1}"]
                        for j, e in enumerate(base.get("path", []))]
                mmr_paths.append({"path": fake, "log_score": -99.0})

        # 推理时 shuffle 路径顺序（与训练时 --shuffle 一致，减少位置偏差）
        import random as _random
        _rng = _random.Random(42)
        _combined = list(enumerate(mmr_paths))
        _rng.shuffle(_combined)
        mmr_paths = [p for _, p in _combined]

        paths_with_meta = [
            (p.get("path", []), p.get("log_score", 0.0), i + 1)
            for i, p in enumerate(mmr_paths)
        ]
        user_content = build_user_content(paths_with_meta, question)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt, mmr_paths, golden, sample

    prepared = [prepare_sample(s) for s in samples]

    # ── 批量推理 ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    results = []
    bs = args.batch_size

    for batch_start in tqdm(range(0, len(prepared), bs),
                            desc=f"Inference (batch={bs})",
                            total=(len(prepared) + bs - 1) // bs):
        batch = prepared[batch_start: batch_start + bs]
        prompts    = [b[0] for b in batch]
        mmr_batch  = [b[1] for b in batch]
        gold_batch = [b[2] for b in batch]
        orig_batch = [b[3] for b in batch]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        raw_texts  = tokenizer.batch_decode(
            output_ids[:, prompt_len:], skip_special_tokens=True
        )

        for raw_text, mmr_paths, golden, orig_sample in zip(
                raw_texts, mmr_batch, gold_batch, orig_batch):

            # 解析输出
            parsed = parse_output(raw_text, args.output_format)

            # 计算 Golden Path 标注（评估忠实度用）
            golden_indices = label_golden_indices(mmr_paths, golden)
            path_entities  = get_all_path_entities(mmr_paths)

            # 答案准确率
            answer_m = compute_answer_metrics(parsed["answers"], golden)

            # 忠实度
            faith_m = compute_faithfulness(
                parsed["cited_indices"], golden_indices,
                parsed["answers"], path_entities,
            )

            rec = {
                **orig_sample,
                "mmr_reason_paths":   mmr_paths,
                "llm_raw_output":     raw_text,
                "llm_pred":           parsed["answers"],
                "cited_indices":      sorted(parsed["cited_indices"]),
                "golden_path_indices": sorted(golden_indices),
                "format_ok":          parsed["format_ok"],
                # 答案准确率
                "hit1":               answer_m["hit1"],
                "hit_any":            answer_m["hit_any"],
                "precision":          round(answer_m["precision"], 4),
                "recall":             round(answer_m["recall"],    4),
                "f1":                 round(answer_m["f1"],        4),
                "exact_match":        answer_m["exact_match"],
                "tp":                 answer_m["tp"],
                "pred_n":             answer_m["pred_n"],
                "gold_n":             answer_m["gold_n"],
                # 忠实度
                "citation_accuracy":   faith_m["citation_accuracy"],
                "citation_recall":     faith_m["citation_recall"],
                "hallucination_rate":  faith_m["hallucination_rate"],
                "hallucinated_entities": faith_m["hallucinated_entities"],
            }
            results.append(rec)

    # ── 写输出 ────────────────────────────────────────────────────────────────
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    log.info("")
    log.info("finish_time: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 60)

    m_all = aggregate(results)
    log_metrics(log, f"ALL  format={args.output_format}", m_all)

    # 按 mmr_answer_path_hit 分层（True=路径命中 / False=路径未命中）
    log.info("")
    log.info("  --- 按路径命中分层（排除路径检索失败的影响）---")
    hit_groups = {"path_hit=True": [], "path_hit=False": []}
    for r in results:
        key = "path_hit=True" if r.get("mmr_answer_path_hit") else "path_hit=False"
        hit_groups[key].append(r)
    for key in ["path_hit=True", "path_hit=False"]:
        items = hit_groups[key]
        if items:
            log_metrics(log, key, aggregate(items))

    # gold_n≤10 子集（排除长尾列表类问题对 Micro 指标的污染）
    short_gold = [r for r in results if r.get("gold_n", 0) <= 10]
    if short_gold and len(short_gold) < len(results):
        log.info("")
        log.info("  --- gold_n≤10 子集（n=%d, 排除长尾列表问题）---", len(short_gold))
        log_metrics(log, "gold_n≤10", aggregate(short_gold))

    # 按 hop 分层
    hop_groups: dict = defaultdict(list)
    for r in results:
        hop_groups[str(r.get("hop", "unknown"))].append(r)

    if len(hop_groups) > 1:
        log.info("")
        log.info("  --- 按跳数分层 ---")
        for hop in sorted(hop_groups.keys()):
            m_hop = aggregate(hop_groups[hop])
            log_metrics(log, f"hop={hop}", m_hop)

    # Format Compliance 细节
    log.info("")
    n_ok  = sum(1 for r in results if r["format_ok"])
    n_all = len(results)
    log.info("  Format Compliance: %d/%d = %.4f", n_ok, n_all, n_ok / n_all if n_all else 0)

    log.info("=" * 60)
    log.info("结果: %s", output_path)
    log.info("日志: %s", log_path)


if __name__ == "__main__":
    main()
