"""精确统计 QLever 滑窗救援五批中：真实答对、盲猜答错、依旧拒答的精确数量。"""

import json

gold_map = {}
with open("data/output/kgqa/ch4_pfit/cwq/rog_transfernet_v1/全量拒答_1ep/seed_17/eval/predictions.jsonl") as f:
    for line in f:
        item = json.loads(line)
        gold_map[item["sample_index"]] = [g.lower() for g in item.get("golden", [])]

preds_file = "data/output/kgqa/ch5_pv_gac/cwq/rog_transfernet_v1/benchmark/qlever_sliding_verified/workflow_full/predictions_no_gold.jsonl"

stats_by_win = {
    "rescue_window_0": {"hit": 0, "hallucinated_wrong": 0, "still_rejected": 0},
    "rescue_window_1": {"hit": 0, "hallucinated_wrong": 0, "still_rejected": 0},
    "rescue_window_2": {"hit": 0, "hallucinated_wrong": 0, "still_rejected": 0},
    "rescue_window_3": {"hit": 0, "hallucinated_wrong": 0, "still_rejected": 0},
    "rescue_window_4": {"hit": 0, "hallucinated_wrong": 0, "still_rejected": 0},
}

total_rejected = 0

with open(preds_file) as f:
    for line in f:
        item = json.loads(line)
        s_idx = item["question"]["sample_index"]
        gold = gold_map.get(s_idx, [])
        is_init_rej = item["initial"]["is_rejection"] or not item["initial"]["answers"]
        if not is_init_rej:
            continue
        total_rejected += 1
        reason = item["reason"]
        if reason not in stats_by_win:
            continue
        final_ans = [a.lower() for a in item["final"]["answers"]]
        final_rej = item["final"]["is_rejection"] or not final_ans

        if final_rej:
            stats_by_win[reason]["still_rejected"] += 1
        else:
            if any(g in final_ans for g in gold):
                stats_by_win[reason]["hit"] += 1
            else:
                stats_by_win[reason]["hallucinated_wrong"] += 1

print(f"初答拒答总数: {total_rejected}")
print("-" * 80)
print(f"窗口批次            | 真实答对(救回) | 盲猜答错(假阳性) | 依旧拒答(安全拦截) | 作答准确率")
print("-" * 80)
total_hit = 0
total_wrong = 0
total_still_rej = 0

for win in sorted(stats_by_win.keys()):
    h = stats_by_win[win]["hit"]
    w = stats_by_win[win]["hallucinated_wrong"]
    r = stats_by_win[win]["still_rejected"]
    total_hit += h
    total_wrong += w
    total_still_rej += r
    attempted = h + w
    prec = (h / attempted * 100) if attempted > 0 else 0.0
    print(f"{win:<19} | {h:<14} | {w:<16} | {r:<18} | {prec:.1f}%")

print("-" * 80)
print(f"五批滑窗合计        | {total_hit:<14} | {total_wrong:<16} | {total_still_rej:<18} | {(total_hit/(total_hit+total_wrong)*100):.1f}%")
