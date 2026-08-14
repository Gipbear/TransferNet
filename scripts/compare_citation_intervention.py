"""引用因果干预配对对比:G0 基线 vs 各干预组(逐样本 sample_index 对齐)。

输入均为 kgqa.pfit.eval 的 predictions.jsonl。输出整体指标差、McNemar、
答案改变率与引用一致性。只做确定性后处理,不调用语言模型。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kgqa.pfit.eval import compute_answer_metrics


def load_predictions(path: str) -> dict:
    """读 predictions.jsonl,返回 {sample_index: rec}。"""
    recs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            idx = r.get("sample_index")
            if idx is not None and idx not in recs:
                recs[idx] = r
    return recs


def overall_metrics(recs: dict) -> dict:
    """从逐样本记录汇总整体指标(与 eval.summarize 口径一致的平均)。"""
    n = len(recs)
    def mean(key):
        return round(sum(r[key] for r in recs.values() if isinstance(r.get(key), (int, float))) / n, 4)

    return {
        "n": n,
        "hit1": mean("hit1"), "hit_any": mean("hit_any"),
        "em": mean("exact_match"),
        "macro_p": mean("precision"), "macro_r": mean("recall"), "macro_f1": mean("f1"),
        "citation_accuracy": mean("citation_accuracy"),
        "citation_recall": mean("citation_recall"),
        "hallucination_rate": mean("hallucination_rate"),
        "format_ok": mean("format_ok"),
        "rejection": mean("is_rejection"),
    }


def mcnemar(g0: dict, gx: dict, key: str = "hit1") -> dict:
    """配对 McNemar:b=G0 对 Gx 错,c=G0 错 Gx 对。返回 b/c/chi2/p。"""
    common = sorted(set(g0) & set(gx))
    b = sum(1 for i in common if g0[i][key] and not gx[i][key])
    c = sum(1 for i in common if not g0[i][key] and gx[i][key])
    if b + c == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p": 1.0, "n": len(common)}
    chi2 = (b - c) ** 2 / (b + c)
    p = math.erfc(math.sqrt(chi2 / 2))
    return {"b": b, "c": c, "chi2": round(chi2, 4), "p": round(p, 4), "n": len(common)}


def answer_change_matrix(g0: dict, gx: dict) -> dict:
    """逐样本答案命中翻转矩阵与答案集合改变率。"""
    common = sorted(set(g0) & set(gx))
    ff = ft = tf = tt = 0
    changed = 0
    for i in common:
        a, b = g0[i]["hit1"], gx[i]["hit1"]
        ff += (not a and not b)
        ft += (not a and b)
        tf += (a and not b)
        tt += (a and b)
        if set(g0[i]["llm_pred"]) != set(gx[i]["llm_pred"]):
            changed += 1
    return {
        "n": len(common),
        "ff_ft_tf_tt": [ff, ft, tf, tt],
        "keep_correct": round(tt / len(common), 4) if common else None,
        "lost_correct": round(tf / len(common), 4) if common else None,
        "answer_changed_rate": round(changed / len(common), 4) if common else None,
    }


def citation_consistency(gx: dict) -> dict:
    """round2 引用与 round1 引用的重叠:只统计干预组(有 path_orig_indices 的记录)。

    overlap_keep:round2 引用中仍属 round1 引用的比例(G1 下即"引用稳定性")。
    """
    total = matched = 0
    total_r1 = overlap = 0
    for r in gx.values():
        r1 = set(r.get("round1_cited_indices", []))
        if not r1:
            continue
        total_r1 += len(r1)
        orig = r.get("path_orig_indices") or []
        for ci in r.get("cited_indices", []):
            total += 1
            if 1 <= ci <= len(orig) and orig[ci - 1] in r1:
                matched += 1
                overlap += 1
    return {
        "round1_cited_total": total_r1,
        "round2_cited_total": total,
        "round2_cites_round1_cited": overlap,
        "round2_cite_is_round1_cited_rate": round(overlap / total, 4) if total else None,
    }


def dropped_path_stats(gx: dict) -> dict:
    """干预组 round2 输入路径数统计:空输入样本占比与平均剩余路径数。

    round1 总路径数无法从干预组记录恢复,故用 G2/G3 的 mean_round2_paths 对齐
    检查删除条数是否配对。
    """
    empty = 0
    n = 0
    n_paths_sum = 0
    n_cited_sum = 0
    for r in gx.values():
        orig = r.get("path_orig_indices") or []
        if not orig:
            empty += 1
        n_paths_sum += len(orig)
        n_cited_sum += len(r.get("round1_cited_indices", []))
        n += 1
    return {
        "n": n,
        "empty_input_samples": empty,
        "mean_round2_paths": round(n_paths_sum / n, 2) if n else None,
        "mean_round1_cited": round(n_cited_sum / n, 2) if n else None,
    }


def stratified_by_reachability(g0: dict, gx: dict) -> dict:
    """按 round2 答案可达性分层:golden 尾实体是否仍在剩余路径中。

    区分两种失效:答案实体被删光(天花板效应)vs 答案仍可达但模型答错
    (模型缺失被引用路径后定位失败)。
    """
    common = sorted(set(g0) & set(gx))
    reach = [i for i in common if gx[i].get("mmr_answer_path_hit")]
    unreach = [i for i in common if not gx[i].get("mmr_answer_path_hit")]

    def hit_mean(idxs, d):
        if not idxs:
            return None
        return round(sum(d[i]["hit1"] for i in idxs) / len(idxs), 4)

    return {
        "reachable_n": len(reach),
        "reachable_hit1_g0": hit_mean(reach, g0),
        "reachable_hit1_gx": hit_mean(reach, gx),
        "unreachable_n": len(unreach),
        "unreachable_hit1_g0": hit_mean(unreach, g0),
        "unreachable_hit1_gx": hit_mean(unreach, gx),
    }


def compare(g0_path: str, group_paths: list, max_samples: int = 0) -> str:
    """主入口:逐组对比并返回报告文本。"""
    g0 = load_predictions(g0_path)
    if max_samples > 0:
        g0 = dict(sorted(g0.items())[:max_samples])
    g0_overall = overall_metrics(g0)
    lines = [f"G0 基线: {g0_path}", f"G0 整体: {json.dumps(g0_overall, ensure_ascii=False)}", ""]
    for gp in group_paths:
        gx = load_predictions(gp)
        if max_samples > 0:
            gx = dict(sorted(gx.items())[:max_samples])
        common = sorted(set(g0) & set(gx))
        if not common:
            lines.append(f"!! {gp}: 与 G0 无共同样本,跳过")
            continue
        gx = {i: gx[i] for i in common}
        g0c = {i: g0[i] for i in common}
        ov = overall_metrics(gx)
        lines.append(f"=== {gp} (共同样本 {len(common)}) ===")
        for key in ("hit1", "hit_any", "em", "macro_p", "macro_r", "macro_f1",
                    "citation_accuracy", "citation_recall", "hallucination_rate",
                    "format_ok", "rejection"):
            lines.append(f"  {key}: G0 {g0_overall[key]} -> {ov[key]} "
                         f"(Δ {round(ov[key] - g0_overall[key], 4)})")
        m = mcnemar(g0c, gx)
        lines.append(f"  McNemar(hit1): b={m['b']} c={m['c']} "
                     f"chi2={m['chi2']} p={m['p']} (n={m['n']})")
        c = answer_change_matrix(g0c, gx)
        lines.append(f"  命中翻转[ff,ft,tf,tt]={c['ff_ft_tf_tt']} "
                     f"保持答对={c['keep_correct']} 丢失答对={c['lost_correct']} "
                     f"答案集改变率={c['answer_changed_rate']}")
        cc = citation_consistency(gx)
        lines.append(f"  引用一致性: round2 引用 {cc['round2_cited_total']} 条,"
                     f"其中属 round1 引用 {cc['round2_cites_round1_cited']} 条"
                     f"({cc['round2_cite_is_round1_cited_rate']})")
        ds = dropped_path_stats(gx)
        lines.append(f"  空输入样本 {ds['empty_input_samples']}/{ds['n']},"
                     f"平均 round1 引用 {ds['mean_round1_cited']} 条,"
                     f"平均 round2 输入路径 {ds['mean_round2_paths']} 条")
        st = stratified_by_reachability(g0c, gx)
        lines.append(f"  按 round2 可达性分层: 可达 {st['reachable_n']} 条"
                     f"(hit1 G0 {st['reachable_hit1_g0']} -> Gx {st['reachable_hit1_gx']}),"
                     f"不可达 {st['unreachable_n']} 条"
                     f"(hit1 G0 {st['unreachable_hit1_g0']} -> Gx {st['unreachable_hit1_gx']})")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--g0", required=True, help="G0 基线 predictions.jsonl")
    p.add_argument("--groups", nargs="+", required=True,
                   help="干预组 predictions.jsonl(可多个)")
    p.add_argument("--max_samples", type=int, default=0,
                   help="只取前 N 条样本对比(0=全部)")
    p.add_argument("--output", default=None, help="报告输出文件(缺省打印到 stdout)")
    a = p.parse_args(argv)
    report = compare(a.g0, a.groups, a.max_samples)
    if a.output:
        Path(a.output).write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
