"""pfit 建集:kgqa.retrieve.cli.retrieve 输出 JSONL → Unsloth SFT messages JSONL。

迁自 llm_infer/build_kgcot_dataset.py。同输入 + 同配置 + 同 seed 时与 legacy
产物 messages 逐条一致(RNG 调用序列严格复刻);数据集差异全部经 PfitDatasetSpec。

用法:
  python -m kgqa.pfit.build --dataset webqsp \\
      --input data/output/kgqa/webqsp/retrieve/train.jsonl \\
      --exp_dir data/output/kgqa/webqsp/pfit/webqsp_main \\
      --format v2 --path_format chain --entity_repr name
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
from collections import Counter
from math import ceil, floor
from typing import Optional

from kgqa.pfit import manifest as manifest_mod
from kgqa.runtime import add_runtime_arguments, configure_runtime, emit_event, update_progress
from kgqa.pfit.formats import (
    build_user_content,
    load_entity_map,
    map_answers,
    select_format_prompt,
)
from kgqa.pfit.specs import get_pfit_spec

log = logging.getLogger("pfit.build")


# ─── Golden Path 标注(与 legacy 行为一致) ─────────────────────────────────────

def label_paths(mmr_reason_paths: list, golden: list) -> list:
    """对每条路径打 Golden/Distractor 标签(尾实体命中 golden 即 Golden)。"""
    golden_set = {g.lower().strip() for g in golden}
    result = []
    for p in mmr_reason_paths:
        edges = p.get("path", [])
        log_score = p.get("log_score", 0.0)
        tail = edges[-1][2].lower().strip() if edges else None
        is_golden = tail in golden_set if tail else False
        result.append((edges, log_score, is_golden))
    return result


def dedupe_labeled_paths_by_tail(labeled: list) -> list:
    """按原始 tail entity 去重,保留首次出现的路径;空 tail 不去重。"""
    result = []
    seen_tails = set()
    for edges, log_score, is_golden in labeled:
        tail = edges[-1][2].lower().strip() if edges else ""
        if tail:
            if tail in seen_tails:
                continue
            seen_tails.add(tail)
        result.append((edges, log_score, is_golden))
    return result


# ─── 数据增强(与 legacy 行为一致) ─────────────────────────────────────────────

def augment(labeled: list, shuffle: bool, distractor_ratio: Optional[float],
            rng: random.Random) -> list:
    """shuffle=完全随机打乱防位置先验;distractor_ratio=干扰路径占比上限。"""
    golden_paths = [p for p in labeled if p[2]]
    distractor_paths = [p for p in labeled if not p[2]]

    if distractor_ratio is not None and 0 < distractor_ratio < 1:
        n_g = len(golden_paths)
        if n_g > 0 and distractor_paths:
            n_d_target = max(1, round(n_g * distractor_ratio / (1 - distractor_ratio)))
            if len(distractor_paths) > n_d_target:
                distractor_paths = rng.sample(distractor_paths, n_d_target)

    result = golden_paths + distractor_paths
    if shuffle:
        rng.shuffle(result)
    return result


# ─── 输出格式(与 legacy 行为一致) ─────────────────────────────────────────────

def output_v1(answers: list) -> str:
    return "Answer: " + " | ".join(answers)


def output_v2(golden_indices: list, answers: list) -> str:
    cited = ", ".join(str(i) for i in sorted(golden_indices))
    return f"Supporting Paths: {cited}\nAnswer: {' | '.join(answers)}"


def output_v2_reject() -> str:
    return "Supporting Paths: None\nAnswer: None"


def output_v3(golden_indices: list, answers: list) -> str:
    return json.dumps(
        {"reasoning": [f"Path {i}" for i in sorted(golden_indices)],
         "answer": answers},
        ensure_ascii=False,
    )


def output_v4(paths_with_meta: list, golden_indices: list, answers: list) -> str:
    golden_set = set(golden_indices)
    relations = list(dict.fromkeys(
        e[1] for edges, _, didx in paths_with_meta
        if didx in golden_set
        for e in edges
    ))
    cited = ", ".join(str(i) for i in sorted(golden_indices))
    rel_str = ", ".join(f'"{r}"' for r in relations[:3])
    reasoning = f"Paths {cited} lead to the answer via {rel_str}."
    return (
        f"Reasoning: {reasoning}\n"
        f"Supporting Paths: {cited}\n"
        f"Answer: {' | '.join(answers)}"
    )


def output_v11(paths_with_meta: list, golden_indices: list, answers: list) -> str:
    golden_set = set(golden_indices)
    reasoning_lines = []
    for edges, log_score, display_idx in paths_with_meta:
        if display_idx in golden_set and edges:
            relations = " -> ".join(f"[{e[1]}]" for e in edges)
            tail = edges[-1][2]
            reasoning_lines.append(f"{display_idx} → {tail} via {relations}")
    reasoning = "\n".join(reasoning_lines) if reasoning_lines else "No supporting path found."
    cited = ", ".join(str(i) for i in sorted(golden_indices))
    return (
        f"[Reasoning]\n{reasoning}\n\n"
        f"[Answer]\n"
        f"Supporting Paths: {cited}\n"
        f"Answer: {' | '.join(answers)}"
    )


# ─── 单样本构造 ────────────────────────────────────────────────────────────────

def make_sample(record: dict, fmt: str, shuffle: bool,
                distractor_ratio: Optional[float], show_score: bool,
                rng: random.Random, *,
                clean_question,
                use_entity_names: bool,
                path_format: str = "arrow",
                entity_map: dict = None,
                include_rejection: bool = False,
                synthetic_rejection: bool = False,
                dedupe_tail_paths: bool = False,
                system_prompt: Optional[str] = None) -> Optional[dict]:
    """从一条 retrieve JSONL 记录构造训练样本;None 表示无效或 Hit@K=0 且未启拒答。

    与 legacy 的差异:问题清洗与 name 措辞由调用方显式传入
    (legacy 用 bool(entity_map) 推断 name 措辞,对天然 name 的数据集会错选 MID 提示词)。
    """
    question = clean_question(record.get("question", ""), record.get("topics", []))
    mmr_paths = record.get("mmr_reason_paths", [])
    golden = record.get("golden", [])

    if not question or not golden:
        return None

    labeled = label_paths(mmr_paths, golden)
    if dedupe_tail_paths:
        labeled = dedupe_labeled_paths_by_tail(labeled)

    has_golden_path = any(is_g for _, _, is_g in labeled)

    def _rejection_sample(paths_labeled, synthetic: bool) -> dict:
        paths_with_meta = [
            (edges, score, i + 1)
            for i, (edges, score, _) in enumerate(paths_labeled)
        ]
        user_content = build_user_content(
            paths_with_meta, question,
            show_score=show_score, path_format=path_format,
            entity_map=entity_map,
        )
        prompt = system_prompt or select_format_prompt(
            "v2", use_entity_names, reject_prompt=True)
        meta = {
            "question":            question,
            "golden":              golden,
            "path_answers":        [],
            "golden_path_indices": [],
            "n_golden":            0,
            "n_distractor":        len(paths_labeled),
            "format":              "v2",
            "show_score":          show_score,
            "path_format":         path_format,
            "entity_map_used":     bool(entity_map),
            "hop":                 record.get("hop"),
            "is_rejection":        True,
        }
        if synthetic:
            meta["synthetic_rejection"] = True
        return {
            "messages": [
                {"role": "system",    "content": prompt},
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": output_v2_reject()},
            ],
            "_meta": meta,
        }

    if synthetic_rejection:
        if not include_rejection or not has_golden_path:
            return None
        distractor_only = [p for p in labeled if not p[2]]
        if not distractor_only:
            return None
        labeled = augment(distractor_only, shuffle, distractor_ratio, rng)
        return _rejection_sample(labeled, synthetic=True)

    # Hit@K=0:路径中无正确答案
    if not has_golden_path:
        if not include_rejection:
            return None
        labeled = augment(labeled, shuffle, distractor_ratio, rng)
        return _rejection_sample(labeled, synthetic=False)

    labeled = augment(labeled, shuffle, distractor_ratio, rng)

    # 重新分配 display index(1-based)
    paths_with_meta = [
        (edges, score, i + 1)
        for i, (edges, score, _) in enumerate(labeled)
    ]
    is_golden_flags = [is_g for _, _, is_g in labeled]
    golden_indices = [i + 1 for i, is_g in enumerate(is_golden_flags) if is_g]

    # 答案使用路径终点实体原文(与路径内容忠实一致);保序去重,极端兜底 golden
    path_answers = list(dict.fromkeys(
        edges[-1][2] for edges, _, is_g in labeled if is_g and edges
    ))
    answer_entities = path_answers if path_answers else golden

    if entity_map:
        answer_entities = map_answers(answer_entities, entity_map)

    user_content = build_user_content(
        paths_with_meta, question,
        show_score=show_score, path_format=path_format,
        entity_map=entity_map,
    )

    if system_prompt is None:
        if include_rejection:
            system_prompt = select_format_prompt("v2", use_entity_names, reject_prompt=True)
        else:
            system_prompt = select_format_prompt(fmt, use_entity_names)

    if fmt == "v1":
        asst = output_v1(answer_entities)
    elif fmt == "v2":
        asst = output_v2(golden_indices, answer_entities)
    elif fmt == "v3":
        asst = output_v3(golden_indices, answer_entities)
    elif fmt == "v4":
        asst = output_v4(paths_with_meta, golden_indices, answer_entities)
    elif fmt == "v11":
        asst = output_v11(paths_with_meta, golden_indices, answer_entities)
    else:
        raise ValueError(f"未知格式: {fmt}")

    return {
        "messages": [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": asst},
        ],
        "_meta": {
            "question":            question,
            "golden":              golden,
            "path_answers":        answer_entities,
            "golden_path_indices": golden_indices,
            "n_golden":            len(golden_indices),
            "n_distractor":        len(labeled) - len(golden_indices),
            "format":              fmt,
            "show_score":          show_score,
            "path_format":         path_format,
            "entity_map_used":     bool(entity_map),
            "hop":                 record.get("hop"),
        },
    }


# ─── 采样 ──────────────────────────────────────────────────────────────────────

def stratified_sample_by_hop(records: list, sample_n: int,
                             rng: random.Random) -> list:
    """按 hop 分层采样,各层配额按原分布比例(最大余数法凑整)。"""
    groups: dict = {}
    for rec in records:
        groups.setdefault(rec.get("hop"), []).append(rec)
    total = len(records)
    quotas = {h: sample_n * len(g) / total for h, g in groups.items()}
    alloc = {h: floor(q) for h, q in quotas.items()}
    remainder = sample_n - sum(alloc.values())
    for h in sorted(quotas, key=lambda h: quotas[h] - alloc[h], reverse=True):
        if remainder <= 0:
            break
        alloc[h] += 1
        remainder -= 1
    sampled = []
    for h in sorted(groups, key=lambda x: (x is None, x)):
        n_h = min(alloc.get(h, 0), len(groups[h]))
        sampled.extend(rng.sample(groups[h], n_h))
    return sampled


# ─── 核心流程 ──────────────────────────────────────────────────────────────────

def _build_samples(records: list, fmt: str, shuffle: bool,
                   distractor_ratio: Optional[float], show_score: bool,
                   rng: random.Random, *, clean_question, use_entity_names: bool,
                   path_format: str, entity_map: Optional[dict],
                   include_rejection: bool, rejection_oversample: int,
                   synthetic_rejection_ratio: float,
                   dedupe_tail_paths: bool,
                   system_prompt: Optional[str] = None) -> tuple[list, dict]:
    """逐条构样 + 拒答上采样/合成(RNG 调用序列与 legacy build 一致)。"""
    common = dict(clean_question=clean_question, use_entity_names=use_entity_names,
                  path_format=path_format, entity_map=entity_map,
                  dedupe_tail_paths=dedupe_tail_paths, system_prompt=system_prompt)

    samples, skipped, n_rejection = [], 0, 0
    rejection_records, synthetic_candidates = [], []

    for rec in records:
        s = make_sample(rec, fmt, shuffle, distractor_ratio, show_score, rng,
                        include_rejection=include_rejection, **common)
        if s is None:
            skipped += 1
        else:
            if s.get("_meta", {}).get("is_rejection"):
                n_rejection += 1
                rejection_records.append(rec)
            else:
                synthetic_candidates.append(rec)
            samples.append(s)

    n_oversampled = 0
    if include_rejection and rejection_oversample > 1 and rejection_records:
        for _ in range(rejection_oversample - 1):
            for rec in rejection_records:
                s = make_sample(rec, fmt, shuffle, distractor_ratio, show_score, rng,
                                include_rejection=True, **common)
                if s is not None:
                    samples.append(s)
                    n_oversampled += 1
        rng.shuffle(samples)

    n_synthetic = 0
    if include_rejection and synthetic_rejection_ratio > 0 and synthetic_candidates:
        if not 0 < synthetic_rejection_ratio < 1:
            raise ValueError("synthetic_rejection_ratio 必须在 0 到 1 之间")
        current_rej = n_rejection + n_oversampled
        target_extra = max(
            0,
            ceil((synthetic_rejection_ratio * len(samples) - current_rej)
                 / (1 - synthetic_rejection_ratio)),
        )
        remaining = target_extra
        while remaining > 0:
            batch = list(synthetic_candidates)
            rng.shuffle(batch)
            produced = 0
            for rec in batch:
                s = make_sample(rec, fmt, shuffle, distractor_ratio, show_score, rng,
                                include_rejection=True, synthetic_rejection=True,
                                **common)
                if s is None:
                    continue
                samples.append(s)
                n_synthetic += 1
                remaining -= 1
                produced += 1
                if remaining == 0:
                    break
            if produced == 0:
                break
        if n_synthetic:
            rng.shuffle(samples)

    stats_extra = {
        "skipped": skipped,
        "n_rejection": n_rejection + n_oversampled + n_synthetic,
        "n_synthetic_rejection": n_synthetic,
    }
    return samples, stats_extra


def _summarize(samples: list, fmt: str, stats_extra: dict) -> dict:
    n = len(samples)
    if n == 0:
        return {"format": fmt, "total": 0, **stats_extra,
                "avg_golden": 0, "avg_distractor": 0}
    avg_golden = round(sum(s["_meta"]["n_golden"] for s in samples) / n, 2)
    avg_distractor = round(sum(s["_meta"]["n_distractor"] for s in samples) / n, 2)
    seq_lens = sorted(
        (len(s["messages"][1]["content"]) + len(s["messages"][2]["content"])) // 4
        for s in samples
    )
    hop_dist = Counter(str(s["_meta"].get("hop", "?")) for s in samples)
    return {
        "format": fmt,
        "total": n,
        **stats_extra,
        "avg_golden": avg_golden,
        "avg_distractor": avg_distractor,
        "seq_len_avg": sum(seq_lens) // n,
        "seq_len_p90": seq_lens[int(n * 0.9)],
        "seq_len_max": seq_lens[-1],
        "hop_dist": dict(hop_dist),
    }


def run_build(*, dataset: str, input_path: str, exp_dir: str, fmt: str,
              path_format: str = "arrow",
              entity_repr: Optional[str] = None,
              entity_map_path: Optional[str] = None,
              shuffle: bool = True, show_score: bool = False,
              distractor_ratio: Optional[float] = None,
              dedupe_tail_paths: bool = False,
              sample_n: int = 0, stratify_by_hop: bool = False,
              include_rejection: bool = False,
              rejection_oversample: int = 1,
              synthetic_rejection_ratio: float = 0.0,
              seed: int = 42,
              output_name: str = "sft_train.jsonl",
              system_prompt_file: str | None = None) -> str:
    """建集主入口。返回产物路径;同配置重跑直接跳过(断点续跑)。"""
    spec = get_pfit_spec(dataset)

    entity_repr = entity_repr or spec.default_entity_repr
    if entity_repr not in spec.entity_reprs:
        raise ValueError(f"{dataset} 不支持 entity_repr={entity_repr!r},可用:{spec.entity_reprs}")
    if include_rejection and not spec.supports_rejection:
        raise ValueError(f"{dataset} 不支持拒答样本构造(检索天花板过高,构造不出有效拒答)")

    system_prompt = None
    if system_prompt_file:
        with open(system_prompt_file, encoding="utf-8") as f:
            system_prompt = f.read().strip()
        if not system_prompt:
            raise ValueError(f"system prompt 文件为空: {system_prompt_file}")

    resolved_map_path = None
    if entity_repr == "name":
        resolved_map_path = entity_map_path or spec.entity_map_path

    # ── manifest 断点续跑判定 ────────────────────────────────────────────────
    out_path = os.path.join(exp_dir, output_name)
    manifest_path = os.path.join(exp_dir, "manifest.json")
    config = {
        "dataset": dataset, "format": fmt, "path_format": path_format,
        "entity_repr": entity_repr, "entity_map_path": resolved_map_path,
        "shuffle": shuffle, "show_score": show_score,
        "distractor_ratio": distractor_ratio,
        "dedupe_tail_paths": dedupe_tail_paths,
        "sample_n": sample_n, "stratify_by_hop": stratify_by_hop,
        "include_rejection": include_rejection,
        "rejection_oversample": rejection_oversample,
        "synthetic_rejection_ratio": synthetic_rejection_ratio,
        "seed": seed, "output_name": output_name,
        "system_prompt": system_prompt,
    }
    inputs = {"retrieve": input_path}
    if resolved_map_path:
        inputs["entity_map"] = resolved_map_path
    section = manifest_mod.make_section(config, inputs)

    existing = manifest_mod.load(manifest_path).get("build")
    if existing is not None:
        if manifest_mod.sections_compatible(existing, section):
            if os.path.isfile(out_path):
                log.info("build 已完成且配置一致,跳过:%s", out_path)
                return out_path
        else:
            raise RuntimeError(
                f"{exp_dir} 已有不同配置的 build 产物;"
                "一目录一实验,请换 exp_dir 或删除旧目录后重跑")

    # ── 读取与校验输入 ───────────────────────────────────────────────────────
    with open(input_path, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]
    missing_golden = sum(1 for r in records if "golden" not in r)
    if missing_golden:
        raise ValueError(
            f"输入缺 golden 字段({missing_golden}/{len(records)} 条);"
            "请用带 golden 输出的 kgqa.retrieve.cli.retrieve 重跑检索")
    log.info("读入 %d 条  dataset=%s fmt=%s path_format=%s entity_repr=%s",
             len(records), dataset, fmt, path_format, entity_repr)

    rng = random.Random(seed)
    if sample_n > 0 and len(records) > sample_n:
        if stratify_by_hop:
            records = stratified_sample_by_hop(records, sample_n, rng)
        else:
            records = rng.sample(records, sample_n)
        log.info("采样后 %d 条(stratify_by_hop=%s)", len(records), stratify_by_hop)

    entity_map = load_entity_map(resolved_map_path) if resolved_map_path else None

    samples, stats_extra = _build_samples(
        records, fmt, shuffle, distractor_ratio, show_score, rng,
        clean_question=spec.clean_question,
        use_entity_names=(entity_repr == "name"),
        path_format=path_format, entity_map=entity_map,
        include_rejection=include_rejection,
        rejection_oversample=rejection_oversample,
        synthetic_rejection_ratio=synthetic_rejection_ratio,
        dedupe_tail_paths=dedupe_tail_paths,
        system_prompt=system_prompt,
    )

    os.makedirs(exp_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    stats = _summarize(samples, fmt, stats_extra)
    section["stats"] = stats
    manifest_mod.merge_section(manifest_path, "build", section)
    log.info("有效样本 %d  丢弃 %d  输出:%s", stats["total"], stats["skipped"], out_path)
    return out_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(description="pfit 建集:retrieve JSONL → SFT JSONL")
    p.add_argument("--dataset", required=True, help="webqsp | metaqa")
    p.add_argument("--input", required=True, help="kgqa.retrieve.cli.retrieve 输出 JSONL")
    p.add_argument("--exp_dir", required=True, help="实验目录(产物+manifest)")
    p.add_argument("--format", default="v2", dest="fmt",
                   choices=["v1", "v2", "v3", "v4", "v11"])
    p.add_argument("--path_format", default="chain",
                   choices=["arrow", "nl", "tuple", "chain"])
    p.add_argument("--entity_repr", default=None, help="mid | name(默认取数据集 spec)")
    p.add_argument("--entity_map", default=None, dest="entity_map_path",
                   help="覆盖 spec 默认的 MID→Name 映射文件")
    p.add_argument("--no_shuffle", action="store_true")
    p.add_argument("--show_score", action="store_true")
    p.add_argument("--distractor_ratio", type=float, default=None)
    p.add_argument("--dedupe_tail_paths", action="store_true")
    p.add_argument("--sample", type=int, default=0, dest="sample_n")
    p.add_argument("--stratify_by_hop", action="store_true",
                   help="按 hop 分层采样(配额按原分布,MetaQA 用)")
    p.add_argument("--include_rejection", action="store_true")
    p.add_argument("--rejection_oversample", type=int, default=1)
    p.add_argument("--synthetic_rejection_ratio", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_name", default="sft_train.jsonl")
    p.add_argument("--system_prompt_file", default=None,
                   help="用文件内容覆盖 system prompt(推理与建集同文,内容进配置指纹)")
    add_runtime_arguments(p)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    run_dir = configure_runtime(a, command="第四章构建训练集", fallback_run_dir=a.exp_dir,
                                manifest={"dataset": a.dataset, "input": a.input})
    output = run_build(dataset=a.dataset, input_path=a.input, exp_dir=a.exp_dir, fmt=a.fmt,
                       path_format=a.path_format, entity_repr=a.entity_repr,
                       entity_map_path=a.entity_map_path,
                       shuffle=not a.no_shuffle, show_score=a.show_score,
                       distractor_ratio=a.distractor_ratio,
                       dedupe_tail_paths=a.dedupe_tail_paths,
                       sample_n=a.sample_n, stratify_by_hop=a.stratify_by_hop,
                       include_rejection=a.include_rejection,
                       rejection_oversample=a.rejection_oversample,
                       synthetic_rejection_ratio=a.synthetic_rejection_ratio,
                       seed=a.seed, output_name=a.output_name,
                       system_prompt_file=a.system_prompt_file)
    update_progress(run_dir, completed=1, total=1, status="completed", phase="构建训练集")
    emit_event(run_dir, "phase_end", phase="构建训练集", output=output)


if __name__ == "__main__":
    main()
