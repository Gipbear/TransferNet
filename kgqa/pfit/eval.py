"""pfit 推理 + 忠实度评测(迁自 llm_infer/eval_faithfulness.py)。

答案指标:Hit@1 / Hit_any / macro+micro P/R/F1 / EM;
忠实度指标:Citation Accuracy / Citation Recall / Hallucination / Format Compliance;
拒答指标:rejection P/R/F1(混淆矩阵)。
数据集分组:spec.group_by_hop=True 时 summary 含 overall + by_hop。

推理三形态:adapter(微调)/ base 零样本(--adapter 缺省)/ --no_paths(无路径基线)。

用法:
  python -m kgqa.pfit.eval --dataset webqsp \\
      --input data/output/kgqa/webqsp/retrieve/test.jsonl \\
      --exp_dir data/output/kgqa/webqsp/pfit/webqsp_main \\
      --adapter data/output/kgqa/webqsp/pfit/webqsp_main/adapter
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
os.environ.setdefault("PYTHONHASHSEED", "0")
import random
import re
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)

from kgqa.pfit import manifest as manifest_mod
from kgqa.pfit.formats import (
    FORMAT_PROMPTS,
    apply_entity_map,
    build_reverse_entity_map,
    build_user_content,
    build_user_content_no_paths,
    select_format_prompt,
)
from kgqa.pfit.specs import get_pfit_spec

log = logging.getLogger("pfit.eval")


def set_global_seed(seed: int = 0) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─── Golden Path 标注(与 legacy 行为一致) ─────────────────────────────────────

def label_golden_indices(mmr_paths: list, golden: list) -> set:
    """返回 1-based display index 集合:尾实体在 golden 中的路径。"""
    golden_set = {g.lower().strip() for g in golden}
    result = set()
    for i, p in enumerate(mmr_paths):
        edges = p.get("path", [])
        tail = edges[-1][2].lower().strip() if edges else None
        if tail and tail in golden_set:
            result.add(i + 1)
    return result


def get_all_path_entities(mmr_paths: list) -> set:
    """收集所有路径中出现过的实体(用于幻觉检测)。"""
    entities = set()
    for p in mmr_paths:
        for edge in p.get("path", []):
            entities.add(edge[0].lower().strip())
            entities.add(edge[2].lower().strip())
    return entities


def dedupe_paths_by_tail(mmr_paths: list) -> list:
    """按原始 tail entity 去重,保留首次出现的路径;空 tail 不去重。"""
    result = []
    seen_tails = set()
    for path_item in mmr_paths:
        edges = path_item.get("path", [])
        tail = edges[-1][2].lower().strip() if edges else ""
        if tail:
            if tail in seen_tails:
                continue
            seen_tails.add(tail)
        result.append(path_item)
    return result


def expand_pred_answers_with_path_constraint(
    pred_answers: list,
    rev_entity_map: dict | None,
    path_mid_entities: set | None,
) -> tuple[list, list]:
    """名称答案先全量展开为候选 MID,再用路径实体做约束消歧。"""
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


# ─── 输出解析(与 legacy 行为一致) ─────────────────────────────────────────────

_ANSWER_RE      = re.compile(r"Answer\s*[:：]\s*(.+)", re.IGNORECASE)
_CITE_RE        = re.compile(r"Supporting\s*Paths?\s*[:：]\s*([\d,\s]+)", re.IGNORECASE)
_JSON_RE        = re.compile(r"\{.*\}", re.DOTALL)
_REJECT_CITE_RE = re.compile(r"Supporting\s*Paths?\s*[:：]\s*\(none\)", re.IGNORECASE)

REJECTION_SENTINEL = "(none)"


def is_rejection_response(parsed: dict) -> bool:
    """所有答案为 (none) 视为主动拒答;空答案视为格式错误。"""
    answers = parsed.get("answers", [])
    if not answers:
        return False
    return all(a.strip().lower() == REJECTION_SENTINEL.lower() for a in answers)


def parse_output(raw: str, fmt: str) -> dict:
    """解析模型输出 → answers / cited_indices(1-based)/ format_ok。"""
    raw = raw.strip()

    def _dedup(lst: list) -> list:
        return list(dict.fromkeys(lst))

    _PLACEHOLDER_RE = re.compile(r"^entity\d*$", re.IGNORECASE)

    def _parse_answers(ans_raw: str) -> list:
        return _dedup(
            e.strip().strip('"\'[]') for e in ans_raw.split("|")
            if e.strip() and not _PLACEHOLDER_RE.match(e.strip().strip('"\'[]'))
        )

    if fmt in ("v0", "v1"):
        m = _ANSWER_RE.search(raw)
        if m:
            ans_raw = m.group(1).strip().splitlines()[0]
            return {"answers": _parse_answers(ans_raw), "cited_indices": set(), "format_ok": True}
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        answers = _parse_answers(lines[-1]) if lines else []
        return {"answers": answers, "cited_indices": set(), "format_ok": False}

    elif fmt == "v2":
        cite_m = _CITE_RE.search(raw)
        reject_cite = bool(_REJECT_CITE_RE.search(raw))
        answer_m = _ANSWER_RE.search(raw)
        format_ok = bool((cite_m or reject_cite) and answer_m)

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
        answer_m = _ANSWER_RE.search(raw)
        answers = _parse_answers(answer_m.group(1).strip().splitlines()[0]) if answer_m else []
        return {"answers": answers, "cited_indices": set(), "format_ok": False}

    elif fmt == "v4":
        cite_m = _CITE_RE.search(raw)
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

    elif fmt == "v11":
        cite_m = _CITE_RE.search(raw)
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


# ─── 指标(与 legacy 行为一致) ─────────────────────────────────────────────────

def norm_entity(s: str) -> str:
    return s.lower().strip()


def compute_answer_metrics(pred: list, gold: list) -> dict:
    pred_set = {norm_entity(e) for e in pred if e.strip()}
    gold_set = {norm_entity(e) for e in gold if e.strip()}

    hit1 = int(bool(pred) and norm_entity(pred[0]) in gold_set)
    hit_any = int(bool(pred_set & gold_set))

    if not pred_set and not gold_set:
        return {"hit1": 1, "hit_any": 1, "precision": 1.0, "recall": 1.0, "f1": 1.0,
                "exact_match": True, "tp": 0, "pred_n": 0, "gold_n": 0}
    if not pred_set or not gold_set:
        return {"hit1": hit1, "hit_any": hit_any, "precision": 0.0, "recall": 0.0,
                "f1": 0.0, "exact_match": False,
                "tp": 0, "pred_n": len(pred_set), "gold_n": len(gold_set)}

    tp = len(pred_set & gold_set)
    p = tp / len(pred_set)
    r = tp / len(gold_set)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"hit1": hit1, "hit_any": hit_any, "precision": p, "recall": r, "f1": f1,
            "exact_match": pred_set == gold_set,
            "tp": tp, "pred_n": len(pred_set), "gold_n": len(gold_set)}


def compute_faithfulness(cited_indices: set, golden_indices: set,
                         pred_answers: list, path_entities: set) -> dict:
    """citation_accuracy / citation_recall / hallucination(拒答哨兵不计幻觉)。"""
    if cited_indices:
        cit_acc = len(cited_indices & golden_indices) / len(cited_indices)
    else:
        cit_acc = 0.0

    if golden_indices:
        cit_rec = len(cited_indices & golden_indices) / len(golden_indices)
    else:
        cit_rec = 0.0

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
        hallu_rate = 0.0

    return {
        "citation_accuracy":   round(cit_acc, 4),
        "citation_recall":     round(cit_rec, 4),
        "hallucination_rate":  round(hallu_rate, 4),
        "hallucinated_entities": hallu_entities,
    }


def compute_rejection_metrics(results: list) -> dict:
    """拒答混淆矩阵与 P/R/F1(path_hit × model_rejected)。"""
    correct_rej = missed_rej = false_rej = correct_ans = 0
    for r in results:
        path_hit = bool(r.get("mmr_answer_path_hit", False))
        model_rejected = bool(r.get("is_rejection", False))

        if path_hit and not model_rejected:
            correct_ans += 1
        elif path_hit and model_rejected:
            false_rej += 1
        elif not path_hit and model_rejected:
            correct_rej += 1
        else:
            missed_rej += 1

    total_rej = correct_rej + false_rej
    unanswerable = correct_rej + missed_rej
    answerable = correct_ans + false_rej

    rej_prec = correct_rej / total_rej if total_rej > 0 else 0.0
    rej_rec = correct_rej / unanswerable if unanswerable > 0 else 0.0
    rej_f1 = (2 * rej_prec * rej_rec / (rej_prec + rej_rec)
              if (rej_prec + rej_rec) > 0 else 0.0)

    return {
        "answerable_n":       answerable,
        "unanswerable_n":     unanswerable,
        "correct_rejections": correct_rej,
        "missed_rejections":  missed_rej,
        "false_rejections":   false_rej,
        "correct_answers":    correct_ans,
        "rejection_precision": round(rej_prec, 4),
        "rejection_recall":    round(rej_rec, 4),
        "rejection_f1":        round(rej_f1, 4),
    }


def aggregate(results: list) -> dict:
    n = len(results)
    if n == 0:
        return {}

    def mean(key):
        return round(sum(r[key] for r in results) / n, 4)

    macro_p = mean("precision")
    macro_r = mean("recall")
    macro_f1 = mean("f1")
    hit1 = mean("hit1")
    hit_any = sum(1 for r in results if r["hit_any"]) / n
    exact = sum(1 for r in results if r["exact_match"]) / n

    tp_sum = sum(r["tp"] for r in results)
    pred_sum = sum(r["pred_n"] for r in results)
    gold_sum = sum(r["gold_n"] for r in results)
    micro_p = tp_sum / pred_sum if pred_sum > 0 else 0.0
    micro_r = tp_sum / gold_sum if gold_sum > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    cit_acc = mean("citation_accuracy")
    cit_rec = mean("citation_recall")
    hallu = mean("hallucination_rate")
    fmt_comp = sum(1 for r in results if r["format_ok"]) / n

    return {
        "n": n,
        "hit1":         round(hit1, 4),
        "hit_any":      round(hit_any, 4),
        "macro_p":      round(macro_p, 4),
        "macro_r":      round(macro_r, 4),
        "macro_f1":     round(macro_f1, 4),
        "micro_p":      round(micro_p, 4),
        "micro_r":      round(micro_r, 4),
        "micro_f1":     round(micro_f1, 4),
        "exact_match":  round(exact, 4),
        "citation_accuracy":  round(cit_acc, 4),
        "citation_recall":    round(cit_rec, 4),
        "hallucination_rate": round(hallu, 4),
        "format_compliance":  round(fmt_comp, 4),
    }


def summarize(results: list, *, group_by_hop: bool) -> dict:
    """overall(+按 hop 分组)汇总;拒答混淆矩阵一并给出。"""
    summary = {"overall": aggregate(results)}
    if group_by_hop:
        groups: dict = {}
        for r in results:
            groups.setdefault(str(r.get("hop", "?")), []).append(r)
        summary["by_hop"] = {h: aggregate(g) for h, g in sorted(groups.items())}
    summary["rejection"] = compute_rejection_metrics(results)
    return summary


def summarize_runs(per_run_summaries: list) -> dict:
    """多轮推理时 overall 指标的 mean±std。"""
    from statistics import mean, pstdev
    keys = [k for k, v in per_run_summaries[0]["overall"].items()
            if isinstance(v, (int, float)) and k != "n"]
    return {
        k: {"mean": round(mean([s["overall"][k] for s in per_run_summaries]), 4),
            "std": round(pstdev([s["overall"][k] for s in per_run_summaries]), 4)}
        for k in keys
    }


# ─── 推理(GPU) ────────────────────────────────────────────────────────────────

def run_single(samples: list, model, tokenizer, cfg: dict, spec,
               run_idx: int, predictions_path: str, num_runs: int) -> list:
    """单轮批量推理;shuffle_paths 时 run_idx 作为 shuffle 偏移。"""
    import torch
    from tqdm import tqdm

    entity_map_dict = cfg["entity_map_dict"]
    rev_entity_map = cfg["rev_entity_map"]
    use_entity_names = cfg["use_entity_names"]
    fmt = cfg["fmt"]
    path_format = cfg["path_format"]
    show_score = cfg["show_score"]

    if cfg["no_paths"]:
        system_prompt = FORMAT_PROMPTS["no_paths"]
    elif cfg["reject_prompt"]:
        system_prompt = select_format_prompt("v2", use_entity_names, reject_prompt=True)
    else:
        system_prompt = select_format_prompt(fmt, use_entity_names)

    def prepare_sample(sample):
        question = spec.clean_question(sample.get("question", ""),
                                       sample.get("topics", []))
        mmr_paths = list(sample.get("mmr_reason_paths", []))
        golden = sample.get("golden", [])

        if cfg["no_paths"]:
            mmr_paths = []
            user_content = build_user_content_no_paths(question)
        else:
            if cfg["noise_paths"] > 0 and mmr_paths:
                existing = list(mmr_paths)
                for i in range(cfg["noise_paths"]):
                    base = existing[i % len(existing)]
                    fake = [[f"noise_{i}_{j}", e[1], f"noise_{i}_{j+1}"]
                            for j, e in enumerate(base.get("path", []))]
                    mmr_paths.append({"path": fake, "log_score": -99.0})

            if cfg["dedupe_tail_paths"]:
                mmr_paths = dedupe_paths_by_tail(mmr_paths)

            if cfg["shuffle_paths"]:
                _seed = (hash(question) + run_idx) % (2 ** 31)
                _rng = random.Random(_seed)
                _rng.shuffle(mmr_paths)

            paths_with_meta = [
                (p.get("path", []), p.get("log_score", 0.0), i + 1)
                for i, p in enumerate(mmr_paths)
            ]
            user_content = build_user_content(
                paths_with_meta, question,
                show_score=show_score, path_format=path_format,
                entity_map=entity_map_dict or None,
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        result = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        input_ids = result["input_ids"] if hasattr(result, "__getitem__") and not isinstance(result, list) else result
        return input_ids, mmr_paths, golden, sample

    prepared = [prepare_sample(s) for s in samples]
    indexed = sorted(enumerate(prepared), key=lambda x: len(x[1][0]))

    results = [None] * len(prepared)
    bs = cfg["batch_size"]
    desc = f"Run {run_idx} / Inference (batch={bs})"

    for batch_start in tqdm(range(0, len(indexed), bs), desc=desc,
                            total=(len(indexed) + bs - 1) // bs):
        batch = indexed[batch_start: batch_start + bs]
        orig_indices = [b[0] for b in batch]
        input_ids_list = [b[1][0] for b in batch]
        mmr_batch = [b[1][1] for b in batch]
        gold_batch = [b[1][2] for b in batch]
        orig_batch = [b[1][3] for b in batch]

        inputs = tokenizer.pad(
            [{"input_ids": ids} for ids in input_ids_list],
            return_tensors="pt",
            padding=True,
            padding_side="left",
        ).to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=cfg["max_new_tokens"],
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        raw_texts = tokenizer.batch_decode(
            output_ids[:, prompt_len:], skip_special_tokens=True
        )

        for orig_idx, raw_text, mmr_paths, golden, orig_sample in zip(
                orig_indices, raw_texts, mmr_batch, gold_batch, orig_batch):

            parsed = parse_output(raw_text, fmt)
            model_rejected = is_rejection_response(parsed)
            golden_indices = label_golden_indices(mmr_paths, golden)
            path_mid_entities = get_all_path_entities(mmr_paths)

            if entity_map_dict:
                path_entities = set()
                for p in mmr_paths:
                    for edge in apply_entity_map(p.get("path", []), entity_map_dict):
                        path_entities.add(edge[0].lower().strip())
                        path_entities.add(edge[2].lower().strip())
            else:
                path_entities = get_all_path_entities(mmr_paths)

            expanded_pred = constrained_pred = None
            if entity_map_dict:
                expanded_pred, constrained_pred = expand_pred_answers_with_path_constraint(
                    pred_answers=parsed["answers"],
                    rev_entity_map=rev_entity_map,
                    path_mid_entities=path_mid_entities,
                )
                answer_m = compute_answer_metrics(constrained_pred, golden)
            else:
                answer_m = compute_answer_metrics(parsed["answers"], golden)

            faith_m = compute_faithfulness(
                parsed["cited_indices"], golden_indices,
                parsed["answers"], path_entities,
            )

            rec = {
                "sample_index":          orig_sample.get("sample_index", -1),
                "question":              orig_sample.get("question", ""),
                "hop":                   orig_sample.get("hop"),
                "golden":                golden,
                "mmr_answer_path_hit":   bool(golden_indices),
                "llm_raw_output":        raw_text,
                "llm_pred":              parsed["answers"],
                "is_rejection":          model_rejected,
                "llm_pred_expanded_mids": expanded_pred if entity_map_dict else None,
                "llm_pred_disambiguated_mids": constrained_pred if entity_map_dict else None,
                "cited_indices":         sorted(parsed["cited_indices"]),
                "golden_path_indices":   sorted(golden_indices),
                "format_ok":             parsed["format_ok"],
                "hit1":                  answer_m["hit1"],
                "hit_any":               answer_m["hit_any"],
                "precision":             round(answer_m["precision"], 4),
                "recall":                round(answer_m["recall"], 4),
                "f1":                    round(answer_m["f1"], 4),
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

    stem, ext = os.path.splitext(predictions_path)
    run_path = predictions_path if num_runs == 1 else f"{stem}_run{run_idx}{ext}"
    with open(run_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return results


def run_eval(*, dataset: str, input_path: str, exp_dir: str,
             adapter: str = None, fmt: str = "v2", path_format: str = "chain",
             entity_repr: str = None, entity_map_path: str = None,
             show_score: bool = False, noise_paths: int = 0,
             dedupe_tail_paths: bool = False, shuffle_paths: bool = False,
             num_runs: int = 1, reject_prompt: bool = False,
             no_paths: bool = False, limit: int = 0,
             model: str = "unsloth/meta-llama-3.1-8b-instruct-bnb-4bit",
             max_seq_length: int = 2048, max_new_tokens: int = 256,
             batch_size: int = 4) -> dict:
    """评测主入口:写 exp_dir/eval/{predictions.jsonl,summary.json};同配置跳过。"""
    spec = get_pfit_spec(dataset)
    entity_repr = entity_repr or spec.default_entity_repr
    if entity_repr not in spec.entity_reprs:
        raise ValueError(f"{dataset} 不支持 entity_repr={entity_repr!r},可用:{spec.entity_reprs}")

    resolved_map_path = None
    if entity_repr == "name":
        resolved_map_path = entity_map_path or spec.entity_map_path

    eval_dir = os.path.join(exp_dir, "eval")
    predictions_path = os.path.join(eval_dir, "predictions.jsonl")
    summary_path = os.path.join(eval_dir, "summary.json")
    manifest_path = os.path.join(exp_dir, "manifest.json")

    config = {
        "dataset": dataset, "adapter": os.path.abspath(adapter) if adapter else None,
        "fmt": fmt, "path_format": path_format, "entity_repr": entity_repr,
        "entity_map_path": resolved_map_path, "show_score": show_score,
        "noise_paths": noise_paths, "dedupe_tail_paths": dedupe_tail_paths,
        "shuffle_paths": shuffle_paths, "num_runs": num_runs,
        "reject_prompt": reject_prompt, "no_paths": no_paths, "limit": limit,
        "model": model, "max_seq_length": max_seq_length,
        "max_new_tokens": max_new_tokens,
    }
    inputs = {"retrieve": input_path}
    if adapter:
        for cand in ("adapter_model.safetensors", "adapter_config.json"):
            p = os.path.join(adapter, cand)
            if os.path.isfile(p):
                inputs["adapter"] = p
                break
    section = manifest_mod.make_section(config, inputs)

    existing = manifest_mod.load(manifest_path).get("eval")
    if existing is not None:
        if manifest_mod.sections_compatible(existing, section) and os.path.isfile(summary_path):
            log.info("eval 已完成且配置一致,跳过:%s", summary_path)
            with open(summary_path, encoding="utf-8") as f:
                return json.load(f)
        if not manifest_mod.sections_compatible(existing, section):
            raise RuntimeError(
                f"{exp_dir} 已有不同配置的 eval 记录;请换 exp_dir 或删除旧目录后重跑")

    set_global_seed(0)

    with open(input_path, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    if limit > 0:
        samples = samples[:limit]
    missing_golden = sum(1 for s in samples if "golden" not in s)
    if missing_golden:
        raise ValueError(
            f"输入缺 golden 字段({missing_golden}/{len(samples)} 条);"
            "请用带 golden 输出的 kgqa.retrieve.cli.retrieve 重跑检索")
    log.info("样本数: %d  adapter=%s  fmt=%s  path_format=%s",
             len(samples), adapter or "None(零样本)", fmt, path_format)

    entity_map_dict = None
    rev_entity_map = None
    if resolved_map_path:
        from kgqa.pfit.formats import load_entity_map
        entity_map_dict = load_entity_map(resolved_map_path)
        rev_entity_map = build_reverse_entity_map(entity_map_dict)

    # ── 加载模型 ─────────────────────────────────────────────────────────────
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        sys.exit("[Error] unsloth 未安装。请运行: pip install unsloth")
    import torch

    model_obj, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
        local_files_only=True,
    )
    if adapter:
        from peft import PeftModel
        model_obj = PeftModel.from_pretrained(model_obj, adapter)
    FastLanguageModel.for_inference(model_obj)
    model_obj.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cfg = {
        "fmt": fmt, "path_format": path_format, "show_score": show_score,
        "noise_paths": noise_paths, "dedupe_tail_paths": dedupe_tail_paths,
        "shuffle_paths": shuffle_paths, "reject_prompt": reject_prompt,
        "no_paths": no_paths, "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "entity_map_dict": entity_map_dict, "rev_entity_map": rev_entity_map,
        "use_entity_names": entity_repr == "name",
    }

    os.makedirs(eval_dir, exist_ok=True)
    per_run_summaries = []
    for run_idx in range(num_runs):
        results = run_single(samples, model_obj, tokenizer, cfg, spec,
                             run_idx, predictions_path, num_runs)
        per_run_summaries.append(summarize(results, group_by_hop=spec.group_by_hop))
        torch.cuda.empty_cache()

    summary = per_run_summaries[0]
    if num_runs > 1:
        summary = {
            "runs": per_run_summaries,
            "mean_std": summarize_runs(per_run_summaries),
            "overall": per_run_summaries[0]["overall"],
        }
    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    manifest_mod.merge_section(manifest_path, "eval", section)
    log.info("summary: %s", summary_path)
    return summary


# ─── CLI ──────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(description="pfit 推理 + 忠实度评测")
    p.add_argument("--dataset", required=True, help="webqsp | metaqa")
    p.add_argument("--input", required=True, help="kgqa.retrieve.cli.retrieve 输出 JSONL(test split)")
    p.add_argument("--exp_dir", required=True, help="实验目录(写 eval/ 与 manifest)")
    p.add_argument("--adapter", default=None, help="LoRA adapter 目录(缺省=base 零样本)")
    p.add_argument("--format", default="v2", dest="fmt",
                   choices=["v0", "v1", "v2", "v3", "v4", "v11"])
    p.add_argument("--path_format", default="chain",
                   choices=["arrow", "nl", "tuple", "chain"])
    p.add_argument("--entity_repr", default=None)
    p.add_argument("--entity_map", default=None, dest="entity_map_path")
    p.add_argument("--show_score", action="store_true")
    p.add_argument("--noise_paths", type=int, default=0)
    p.add_argument("--dedupe_tail_paths", action="store_true")
    p.add_argument("--shuffle_paths", action="store_true")
    p.add_argument("--num_runs", type=int, default=1)
    p.add_argument("--reject_prompt", action="store_true")
    p.add_argument("--no_paths", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--model", default="unsloth/meta-llama-3.1-8b-instruct-bnb-4bit")
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=4)
    return p


def main(argv=None):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    a = build_parser().parse_args(argv)
    summary = run_eval(
        dataset=a.dataset, input_path=a.input, exp_dir=a.exp_dir,
        adapter=a.adapter, fmt=a.fmt, path_format=a.path_format,
        entity_repr=a.entity_repr, entity_map_path=a.entity_map_path,
        show_score=a.show_score, noise_paths=a.noise_paths,
        dedupe_tail_paths=a.dedupe_tail_paths, shuffle_paths=a.shuffle_paths,
        num_runs=a.num_runs, reject_prompt=a.reject_prompt, no_paths=a.no_paths,
        limit=a.limit, model=a.model, max_seq_length=a.max_seq_length,
        max_new_tokens=a.max_new_tokens, batch_size=a.batch_size)
    print(json.dumps(summary.get("overall", summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
