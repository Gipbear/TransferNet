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

# kg_format 与本脚本同目录（llm_infer/）
from kg_format import (FORMAT_PROMPTS, build_user_content, build_user_content_no_paths,
                       load_entity_map, apply_entity_map, build_reverse_entity_map)


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


def expand_pred_answers_with_path_constraint(
    pred_answers: list,
    rev_entity_map: dict | None,
    path_mid_entities: set | None,
) -> tuple[list, list]:
    """名称答案先全量展开，再用路径实体 MID 做约束消歧。

    返回:
      expanded_pred:      原始 name -> all candidate MIDs 展开结果
      constrained_pred:   若候选 MID 与路径实体有交集，则仅保留交集；否则回退到原展开
    """
    expanded_pred = []
    constrained_pred = []
    path_mid_entities = path_mid_entities or set()

    for answer in pred_answers:
        key = answer.lower().strip()
        if rev_entity_map and key in rev_entity_map:
            expanded = sorted(rev_entity_map[key])
            constrained = [
                mid for mid in expanded
                if mid.lower().strip() in path_mid_entities
            ]
            expanded_pred.extend(expanded)
            constrained_pred.extend(constrained if constrained else expanded)
        else:
            expanded_pred.append(answer)
            constrained_pred.append(answer)

    return expanded_pred, constrained_pred


# ─── 输出解析 ──────────────────────────────────────────────────────────────────

_ANSWER_RE      = re.compile(r"Answer\s*[:：]\s*(.+)", re.IGNORECASE)
_CITE_RE        = re.compile(r"Supporting\s*Paths?\s*[:：]\s*([\d,\s]+)", re.IGNORECASE)
_JSON_RE        = re.compile(r"\{.*\}", re.DOTALL)
_REJECT_CITE_RE = re.compile(r"Supporting\s*Paths?\s*[:：]\s*\(none\)", re.IGNORECASE)

REJECTION_SENTINEL = "(none)"


def is_rejection_response(parsed: dict) -> bool:
    """检查模型输出是否为拒答响应（所有答案为 (none) 或答案为空）。"""
    answers = parsed.get("answers", [])
    if not answers:
        return False  # 空答案视为格式错误，不视为主动拒答
    return all(a.strip().lower() == REJECTION_SENTINEL.lower() for a in answers)


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
        # 拒答格式: "Supporting Paths: (none)\nAnswer: (none)"
        cite_m      = _CITE_RE.search(raw)
        reject_cite = bool(_REJECT_CITE_RE.search(raw))
        answer_m    = _ANSWER_RE.search(raw)
        format_ok   = bool((cite_m or reject_cite) and answer_m)

        cited_indices = set()
        if cite_m and not reject_cite:
            for tok in re.split(r"[,\s]+", cite_m.group(1)):
                tok = tok.strip()
                if tok.isdigit():
                    cited_indices.add(int(tok))

        if reject_cite and answer_m and REJECTION_SENTINEL in answer_m.group(1).lower():
            answers = [REJECTION_SENTINEL]
        else:
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
        # 期望: "Reasoning: ...\nSupporting Paths: ...\nAnswer: ..."
        cite_m   = _CITE_RE.search(raw)
        answer_m = _ANSWER_RE.search(raw)
        has_reasoning = bool(re.search(r"Reasoning\s*[:：]", raw, re.IGNORECASE))
        format_ok = bool(cite_m and answer_m and has_reasoning)

        cited_indices = set()
        if cite_m:
            for tok in re.split(r"[,\s]+", cite_m.group(1)):
                tok = tok.strip()
                if tok.isdigit():
                    cited_indices.add(int(tok))

        answers = _parse_answers(answer_m.group(1).strip().splitlines()[0]) if answer_m else []
        return {"answers": answers, "cited_indices": cited_indices, "format_ok": format_ok}

    elif fmt == "v5":
        # V5 输出格式与 V2 完全相同（仅输入路径为自然语言格式）
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

    elif fmt == "v11":
        # V11 Full CoT（备用）: [Reasoning]...[Answer]\nSupporting Paths: ...\nAnswer: ...
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
    effective_pred_answers = [
        e for e in pred_answers
        if norm_entity(e) != norm_entity(REJECTION_SENTINEL)
    ]
    if effective_pred_answers:
        hallu_entities = [e for e in effective_pred_answers
                          if norm_entity(e) not in path_entities]
        hallu_rate = len(hallu_entities) / len(effective_pred_answers)
    else:
        hallu_entities = []
        hallu_rate     = 0.0

    return {
        "citation_accuracy":   round(cit_acc, 4),
        "citation_recall":     round(cit_rec, 4),
        "hallucination_rate":  round(hallu_rate, 4),
        "hallucinated_entities": hallu_entities,
    }


# ─── 拒答指标 ─────────────────────────────────────────────────────────────────

def compute_rejection_metrics(results: list) -> dict:
    """计算拒答能力的混淆矩阵和 P/R/F1。

    四类情形：
      correct_rejections:  模型拒答 & path_hit=False（正确拒答，TN）
      missed_rejections:   模型回答 & path_hit=False（漏拒，FP）
      false_rejections:    模型拒答 & path_hit=True（误拒，FN）
      correct_answers:     模型回答 & path_hit=True（正常作答）

    Rejection Precision = correct_rej / (correct_rej + false_rej)
    Rejection Recall    = correct_rej / (correct_rej + missed_rej)
    """
    correct_rej = missed_rej = false_rej = correct_ans = 0
    for r in results:
        path_hit      = bool(r.get("mmr_answer_path_hit", False))
        model_rejected = bool(r.get("is_rejection", False))

        if path_hit and not model_rejected:
            correct_ans += 1
        elif path_hit and model_rejected:
            false_rej += 1
        elif not path_hit and model_rejected:
            correct_rej += 1
        else:
            missed_rej += 1

    total_rej    = correct_rej + false_rej
    unanswerable = correct_rej + missed_rej
    answerable   = correct_ans + false_rej

    rej_prec = correct_rej / total_rej    if total_rej    > 0 else 0.0
    rej_rec  = correct_rej / unanswerable if unanswerable > 0 else 0.0
    rej_f1   = (2 * rej_prec * rej_rec / (rej_prec + rej_rec)
                if (rej_prec + rej_rec) > 0 else 0.0)

    return {
        "answerable_n":       answerable,
        "unanswerable_n":     unanswerable,
        "correct_rejections": correct_rej,
        "missed_rejections":  missed_rej,
        "false_rejections":   false_rej,
        "correct_answers":    correct_ans,
        "rejection_precision": round(rej_prec, 4),
        "rejection_recall":    round(rej_rec,  4),
        "rejection_f1":        round(rej_f1,   4),
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


def log_metrics_with_std(log: logging.Logger, runs_agg: list):
    """跨多轮汇总：输出 mean ± std（num_runs > 1 时调用）。"""
    import math

    def mean_std(key):
        vals = [m[key] for m in runs_agg if m.get(key) is not None]
        if not vals:
            return 0.0, 0.0
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / len(vals)
        return mu, math.sqrt(var)

    n_runs = len(runs_agg)
    avg_n  = round(sum(m["n"] for m in runs_agg) / n_runs)
    log.info("  [ 多轮汇总  num_runs=%d ]  (avg n=%d)", n_runs, avg_n)

    h1_mu, h1_sd = mean_std("hit1")
    ha_mu, ha_sd = mean_std("hit_any")
    log.info("    Hit@1             : %.4f ± %.4f", h1_mu, h1_sd)
    log.info("    Hit@Any           : %.4f ± %.4f", ha_mu, ha_sd)

    mp_mu, mp_sd = mean_std("macro_p")
    mr_mu, mr_sd = mean_std("macro_r")
    mf_mu, mf_sd = mean_std("macro_f1")
    log.info("    Macro  P/R/F1     : %.4f±%.4f / %.4f±%.4f / %.4f±%.4f",
             mp_mu, mp_sd, mr_mu, mr_sd, mf_mu, mf_sd)

    up_mu, up_sd = mean_std("micro_p")
    ur_mu, ur_sd = mean_std("micro_r")
    uf_mu, uf_sd = mean_std("micro_f1")
    log.info("    Micro  P/R/F1     : %.4f±%.4f / %.4f±%.4f / %.4f±%.4f",
             up_mu, up_sd, ur_mu, ur_sd, uf_mu, uf_sd)

    em_mu, em_sd = mean_std("exact_match")
    log.info("    Exact Match       : %.4f ± %.4f", em_mu, em_sd)

    ca_mu, ca_sd = mean_std("citation_accuracy")
    cr_mu, cr_sd = mean_std("citation_recall")
    log.info("    Citation Accuracy : %.4f ± %.4f", ca_mu, ca_sd)
    log.info("    Citation Recall   : %.4f ± %.4f", cr_mu, cr_sd)

    hl_mu, hl_sd = mean_std("hallucination_rate")
    fc_mu, fc_sd = mean_std("format_compliance")
    log.info("    Hallucination Rate: %.4f ± %.4f", hl_mu, hl_sd)
    log.info("    Format Compliance : %.4f ± %.4f", fc_mu, fc_sd)


def log_rejection_metrics_with_std(log: logging.Logger, runs_results: list):
    """跨多轮输出拒答指标 mean±std。"""
    import math

    rejection_runs = [
        compute_rejection_metrics(results)
        for results in runs_results
        if any(r.get("is_rejection", False) for r in results)
        or any(not r.get("mmr_answer_path_hit", True) for r in results)
    ]
    if not rejection_runs:
        return

    def mean_std(key):
        vals = [m[key] for m in rejection_runs]
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / len(vals)
        return mu, math.sqrt(var)

    log.info("")
    log.info("  --- Rejection Analysis 多轮汇总 ---")
    for key, label in [
        ("correct_rejections", "Correct Rejections"),
        ("missed_rejections",  "Missed  Rejections"),
        ("false_rejections",   "False   Rejections"),
        ("rejection_precision", "Rejection Precision"),
        ("rejection_recall",    "Rejection Recall"),
        ("rejection_f1",        "Rejection F1"),
    ]:
        mu, sd = mean_std(key)
        log.info("    %-31s: %.4f ± %.4f", label, mu, sd)


# ─── 主函数 ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="第四章忠实度评估")
    p.add_argument("--input",         required=True, help="predict JSONL（含 mmr_reason_paths / golden）")
    p.add_argument("--output",        default=None,  help="输出目录（文件名自动生成）")
    p.add_argument("--model",         default="unsloth/meta-llama-3.1-8b-instruct-bnb-4bit")
    p.add_argument("--adapter",       default=None,  help="LoRA adapter 目录（微调模型）")
    p.add_argument("--output_format", default="v2",
                   choices=["v0", "v1", "v2", "v3", "v4", "v5", "v11"],
                   help="模型输出格式（决定 prompt 和解析方式）")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--batch_size",     type=int, default=8,
                   help="批量推理大小（越大越快，受显存限制；建议 4~16）")
    p.add_argument("--limit",          type=int, default=0, help="只评估前 N 条（0=全部）")
    p.add_argument("--noise_paths",    type=int, default=0,
                   help="推理时在输入末尾追加 N 条伪干扰路径（抗噪鲁棒性实验）")
    p.add_argument("--show_score",      action="store_true",
                   help="路径字符串中包含 [score=S]（默认不含；需与训练时对齐）")
    p.add_argument("--path_format",    default="arrow",
                   choices=["arrow", "nl", "tuple", "chain"],
                   help="路径表示方式: arrow=符号格式(默认) nl=自然语言格式 tuple=三元组 chain=连续链式")
    p.add_argument("--entity_map", default=None,
                   help="实体映射文件路径 (MID→Name, tab-separated)")
    p.add_argument("--num_runs",       type=int, default=1,
                   help="多轮 shuffle 推理次数（每轮使用不同路径排列顺序），"
                        "num_runs>1 时日志输出 mean±std（默认 1，向后兼容）")
    p.add_argument("--reject_prompt", action="store_true",
                   help="使用含拒答规则的 system prompt（Group F）")
    p.add_argument("--no_paths", action="store_true",
                   help="忽略输入中的检索路径，直接以问题裸文本推理（Group H）")
    return p.parse_args()


def resolve_output(input_path: str, output_arg, fmt: str, adapter: str,
                   no_paths: bool = False) -> str:
    stem   = os.path.splitext(os.path.basename(input_path))[0]
    suffix = f"_{fmt}"
    if no_paths:
        suffix += "_nopaths"
    if adapter:
        suffix += "_ft"
    suffix += "_eval.jsonl"
    outdir = output_arg if output_arg else os.path.dirname(os.path.abspath(input_path))
    return os.path.join(outdir, stem + suffix)


def run_single(samples: list, model, tokenizer, args, log: logging.Logger,
               run_idx: int, output_path: str) -> list:
    """
    单轮推理：用 run_idx 作为 shuffle 偏移量，返回样本级指标列表。
    run_idx=0 时种子为 hash(question) % 2**31（与原始行为一致）。
    结果同时写入 {output_path_stem}_run{run_idx}.jsonl。
    """
    import random as _random

    entity_map_dict = getattr(args, "entity_map_dict", None)
    rev_entity_map = getattr(args, "rev_entity_map", None)
    use_reject_prompt = getattr(args, "reject_prompt", False)
    use_no_paths      = getattr(args, "no_paths", False)
    if use_no_paths:
        # Group H: 无路径输入，使用专用 system prompt
        system_prompt = FORMAT_PROMPTS["no_paths"]
    elif use_reject_prompt:
        # Group F: 使用含拒答规则的 system prompt
        if entity_map_dict:
            system_prompt = FORMAT_PROMPTS["v2_name_reject"]
        else:
            system_prompt = FORMAT_PROMPTS["v2_reject"]
    elif entity_map_dict and args.output_format == "v2":
        system_prompt = FORMAT_PROMPTS["v2_name"]
    else:
        system_prompt = FORMAT_PROMPTS[args.output_format]
        if entity_map_dict and args.output_format not in ("v2",):
            log.warning(
                "entity_map 已启用，但 output_format=%s 没有对应的 _name system prompt。"
                "系统 prompt 仍要求输出实体 ID，而路径已替换为实体名称，可能导致模型混淆。"
                "建议使用 --output_format v2 配合 --entity_map。",
                args.output_format,
            )
    show_score    = args.show_score
    path_format   = getattr(args, "path_format", "arrow")
    # V5 若未显式指定，自动切换为自然语言路径
    if args.output_format == "v5" and path_format == "arrow":
        path_format = "nl"

    def prepare_sample(sample):
        question  = sample.get("question", "")
        mmr_paths = list(sample.get("mmr_reason_paths", []))
        golden    = sample.get("golden", [])

        if use_no_paths:
            # Group H: 丢弃所有检索路径，仅以问题本身作为输入
            mmr_paths = []
            user_content = build_user_content_no_paths(question)
        else:
            if args.noise_paths > 0 and mmr_paths:
                existing = list(mmr_paths)
                for i in range(args.noise_paths):
                    base = existing[i % len(existing)]
                    fake = [[f"noise_{i}_{j}", e[1], f"noise_{i}_{j+1}"]
                            for j, e in enumerate(base.get("path", []))]
                    mmr_paths.append({"path": fake, "log_score": -99.0})

            # run_idx=0 时种子 = hash(question) % 2**31，与原始行为完全一致
            # run_idx>0 时加偏移，产生不同的排列顺序
            _seed = (hash(question) + run_idx) % (2 ** 31)
            _rng  = _random.Random(_seed)
            _rng.shuffle(mmr_paths)

            paths_with_meta = [
                (p.get("path", []), p.get("log_score", 0.0), i + 1)
                for i, p in enumerate(mmr_paths)
            ]
            user_content = build_user_content(
                paths_with_meta, question,
                show_score=show_score, path_format=path_format,
                entity_map=entity_map_dict,
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ]
        result = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        # 新版 transformers 可能返回 BatchEncoding，取 input_ids list
        input_ids = result["input_ids"] if hasattr(result, "__getitem__") and not isinstance(result, list) else result
        return input_ids, mmr_paths, golden, sample

    prepared = [prepare_sample(s) for s in samples]
    # 按 prompt token 数升序排序；orig_idx 用于还原输出顺序
    indexed = sorted(enumerate(prepared), key=lambda x: len(x[1][0]))

    results = [None] * len(prepared)
    bs = args.batch_size
    desc = f"Run {run_idx} / Inference (batch={bs})"

    for batch_start in tqdm(range(0, len(indexed), bs), desc=desc,
                            total=(len(indexed) + bs - 1) // bs):
        batch          = indexed[batch_start: batch_start + bs]
        orig_indices   = [b[0] for b in batch]
        input_ids_list = [b[1][0] for b in batch]
        mmr_batch      = [b[1][1] for b in batch]
        gold_batch     = [b[1][2] for b in batch]
        orig_batch     = [b[1][3] for b in batch]

        inputs = tokenizer.pad(
            [{"input_ids": ids} for ids in input_ids_list],
            return_tensors="pt",
            padding=True,
            padding_side="left",
        ).to(model.device)

        with torch.inference_mode():
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

        for orig_idx, raw_text, mmr_paths, golden, orig_sample in zip(
                orig_indices, raw_texts, mmr_batch, gold_batch, orig_batch):

            parsed         = parse_output(raw_text, args.output_format)
            model_rejected = is_rejection_response(parsed)
            golden_indices = label_golden_indices(mmr_paths, golden)
            path_mid_entities = get_all_path_entities(mmr_paths)

            # For hallucination: collect path entities in the format used during inference
            if entity_map_dict:
                path_entities = set()
                for p in mmr_paths:
                    for edge in apply_entity_map(p.get("path", []), entity_map_dict):
                        path_entities.add(edge[0].lower().strip())
                        path_entities.add(edge[2].lower().strip())
            else:
                path_entities = get_all_path_entities(mmr_paths)

            # For answer metrics: expand predicted names to MIDs when entity_map active
            expanded_pred = None
            constrained_pred = None
            if entity_map_dict:
                expanded_pred, constrained_pred = expand_pred_answers_with_path_constraint(
                    pred_answers=parsed["answers"],
                    rev_entity_map=rev_entity_map,
                    path_mid_entities=path_mid_entities,
                )
                answer_m = compute_answer_metrics(constrained_pred, golden)
            else:
                answer_m = compute_answer_metrics(parsed["answers"], golden)

            faith_m        = compute_faithfulness(
                parsed["cited_indices"], golden_indices,
                parsed["answers"], path_entities,
            )

            rec = {
                **orig_sample,
                "mmr_reason_paths":      mmr_paths,
                "llm_raw_output":        raw_text,
                "llm_pred":              parsed["answers"],
                "is_rejection":          model_rejected,
                # 当 entity_map 启用时，保留原始全量展开与路径约束消歧后的 MID 列表
                "llm_pred_expanded_mids": expanded_pred if entity_map_dict else None,
                "llm_pred_disambiguated_mids": constrained_pred if entity_map_dict else None,
                "cited_indices":         sorted(parsed["cited_indices"]),
                "golden_path_indices":   sorted(golden_indices),
                "format_ok":             parsed["format_ok"],
                "hit1":                  answer_m["hit1"],
                "hit_any":               answer_m["hit_any"],
                "precision":             round(answer_m["precision"], 4),
                "recall":                round(answer_m["recall"],    4),
                "f1":                    round(answer_m["f1"],        4),
                "exact_match":           answer_m["exact_match"],
                "tp":                    answer_m["tp"],
                "pred_n":                answer_m["pred_n"],
                "gold_n":                answer_m["gold_n"],
                "citation_accuracy":     faith_m["citation_accuracy"],
                "citation_recall":       faith_m["citation_recall"],
                "hallucination_rate":    faith_m["hallucination_rate"],
                "hallucinated_entities": faith_m["hallucinated_entities"],
            }
            results[orig_idx] = rec

    # 每轮结果写入独立文件（num_runs=1 时写 output_path 本身，保持原有行为）
    stem, ext = os.path.splitext(output_path)
    run_path = output_path if args.num_runs == 1 else f"{stem}_run{run_idx}{ext}"
    with open(run_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if args.num_runs > 1:
        log.info("Run %d 结果: %s", run_idx, run_path)

    return results


def _log_stratified(log: logging.Logger, results: list):
    """输出分层汇总（路径命中 / gold_n≤10 / hop），复用于单轮和多轮。"""
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

    short_gold = [r for r in results if r.get("gold_n", 0) <= 10]
    if short_gold and len(short_gold) < len(results):
        log.info("")
        log.info("  --- gold_n≤10 子集（n=%d, 排除长尾列表问题）---", len(short_gold))
        log_metrics(log, "gold_n≤10", aggregate(short_gold))

    hop_groups: dict = defaultdict(list)
    for r in results:
        hop_groups[str(r.get("hop", "unknown"))].append(r)
    if len(hop_groups) > 1:
        log.info("")
        log.info("  --- 按跳数分层 ---")
        for hop in sorted(hop_groups.keys()):
            log_metrics(log, f"hop={hop}", aggregate(hop_groups[hop]))

    n_ok  = sum(1 for r in results if r["format_ok"])
    n_all = len(results)
    log.info("")
    log.info("  Format Compliance: %d/%d = %.4f", n_ok, n_all,
             n_ok / n_all if n_all else 0)

    # 拒答分析（当存在拒答样本或存在 path_hit=False 样本时输出）
    has_rejection = any(r.get("is_rejection", False) for r in results)
    has_unanswerable = any(not r.get("mmr_answer_path_hit", True) for r in results)
    if has_rejection or has_unanswerable:
        rej_m = compute_rejection_metrics(results)
        log.info("")
        log.info("  --- Rejection Analysis ---")
        log.info("    Answerable   (path_hit=True)  : %d", rej_m["answerable_n"])
        log.info("    Unanswerable (path_hit=False)  : %d", rej_m["unanswerable_n"])
        log.info("    Correct Rejections             : %d", rej_m["correct_rejections"])
        log.info("    Missed  Rejections             : %d", rej_m["missed_rejections"])
        log.info("    False   Rejections             : %d", rej_m["false_rejections"])
        log.info("    Correct Answers                : %d", rej_m["correct_answers"])
        log.info("    Rejection Precision            : %.4f", rej_m["rejection_precision"])
        log.info("    Rejection Recall               : %.4f", rej_m["rejection_recall"])
        log.info("    Rejection F1                   : %.4f", rej_m["rejection_f1"])
        # 仅在可回答且模型作答的子集上统计答案质量
        answerable_answered = [r for r in results
                               if r.get("mmr_answer_path_hit") and not r.get("is_rejection")]
        if answerable_answered and len(answerable_answered) < len(results):
            log.info("")
            log.info("  --- Answerable & Answered subset (n=%d) ---", len(answerable_answered))
            log_metrics(log, "answerable_answered", aggregate(answerable_answered))


def main():
    args = parse_args()

    output_path = resolve_output(args.input, args.output, args.output_format, args.adapter,
                                 no_paths=getattr(args, "no_paths", False))
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
    log.info("  show_score    : %s", args.show_score)
    log.info("  path_format   : %s", getattr(args, 'path_format', 'arrow'))
    log.info("  entity_map    : %s", args.entity_map or "None")
    log.info("  num_runs      : %d", args.num_runs)
    log.info("=" * 60)

    # ── 加载模型（只加载一次）────────────────────────────────────────────────
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        sys.exit("[Error] unsloth 未安装。请运行: pip install unsloth")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
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

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 读取数据 ──────────────────────────────────────────────────────────────
    with open(args.input, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    if args.limit > 0:
        samples = samples[:args.limit]
    log.info("样本数: %d", len(samples))

    args.entity_map_dict = None
    args.rev_entity_map = None
    if args.entity_map:
        args.entity_map_dict = load_entity_map(args.entity_map)
        args.rev_entity_map = build_reverse_entity_map(args.entity_map_dict)
        log.info("加载实体映射: %s (%d 条, 反向映射 %d 键)",
                 args.entity_map, len(args.entity_map_dict), len(args.rev_entity_map))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # ── 多轮推理 ──────────────────────────────────────────────────────────────
    results_per_run = []
    for run_idx in range(args.num_runs):
        if args.num_runs > 1:
            log.info("")
            log.info("--- Run %d / %d ---", run_idx, args.num_runs - 1)
        results = run_single(samples, model, tokenizer, args, log, run_idx, output_path)
        results_per_run.append(results)
        torch.cuda.empty_cache()

    log.info("")
    log.info("finish_time: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 60)

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    if args.num_runs == 1:
        # 单轮：保持原有输出格式，完全向后兼容
        results = results_per_run[0]
        m_all = aggregate(results)
        log_metrics(log, f"ALL  format={args.output_format}", m_all)
        _log_stratified(log, results)
    else:
        # 多轮：先输出每轮概要，再输出跨轮 mean±std
        runs_agg = []
        for run_idx, results in enumerate(results_per_run):
            m = aggregate(results)
            runs_agg.append(m)
            log_metrics(log, f"ALL  format={args.output_format}  run={run_idx}", m)

        log.info("")
        log.info("  --- 多轮汇总 (num_runs=%d) ---", args.num_runs)
        log_metrics_with_std(log, runs_agg)
        log_rejection_metrics_with_std(log, results_per_run)

        # 分层统计基于最后一轮结果（代表性），不逐轮重复
        log.info("")
        log.info("  --- 分层统计（以 run=0 为代表）---")
        _log_stratified(log, results_per_run[0])

    log.info("=" * 60)
    log.info("结果: %s", output_path)
    log.info("日志: %s", log_path)


if __name__ == "__main__":
    main()
