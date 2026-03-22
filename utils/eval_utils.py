"""
共享 MMR 评测工具，供 WebQSP / MetaQA_KB 等数据集复用。
不依赖任何数据集特定逻辑，只依赖 utils/path_utils.py 和标准库。

包含：
  create_mmr_stats      - 初始化 MMR 统计 dict
  create_thresh_stats   - 初始化多阈值统计 dict
  create_std_stats      - 初始化标准束搜索对比统计 dict
  update_mmr_stats      - 累加单样本 MMR 路径指标 + 多样性指标
  update_thresh_stats   - 累加单样本多阈值统计
  update_std_stats      - 累加单样本标准束搜索指标
  print_validate_results - 打印完整评测报告
"""

from utils.path_utils import compute_path_metrics, compute_path_diversity


def create_mmr_stats():
    return {"count": 0, "answer_path_hit": 0, "top1_hit": 0,
            "recall_sum": 0.0, "precision_sum": 0.0, "f1_sum": 0.0,
            "jaccard_div_sum": 0.0, "tail_div_sum": 0.0, "edge_cov_sum": 0.0}


def create_thresh_stats(thresholds):
    return {
        t: {"correct": 0, "total": 0,
            "precision_sum": 0.0, "recall_sum": 0.0, "f1_sum": 0.0}
        for t in thresholds
    }


def create_std_stats():
    return {"answer_path_hit": 0, "top1_hit": 0, "recall_sum": 0.0,
            "precision_sum": 0.0, "f1_sum": 0.0, "count": 0}


def update_mmr_stats(mmr_stats, mmr_paths, gold_ids):
    """累加单样本 MMR 路径检索指标和多样性指标，返回 (path_metrics, diversity_metrics)。"""
    m = compute_path_metrics(mmr_paths, gold_ids)
    d = compute_path_diversity(mmr_paths)
    mmr_stats["count"]           += 1
    mmr_stats["answer_path_hit"] += int(m["answer_hit"])
    mmr_stats["top1_hit"]        += int(m["top1_hit"])
    mmr_stats["recall_sum"]      += m["recall"]
    mmr_stats["precision_sum"]   += m["precision"]
    mmr_stats["f1_sum"]          += m["f1"]
    mmr_stats["jaccard_div_sum"] += d["jaccard_diversity"]
    mmr_stats["tail_div_sum"]    += d["tail_diversity"]
    mmr_stats["edge_cov_sum"]    += d["edge_coverage"]
    return m, d


def update_thresh_stats(thresh_stats, e_score_i, gold_ids, thresholds):
    """累加单样本多阈值 e_score 统计。e_score_i 为 1-D CPU tensor。"""
    for t in thresholds:
        pred_ids = set(e_score_i.gt(t).nonzero().squeeze(1).tolist())
        tp = len(pred_ids & gold_ids)
        thr_p = tp / len(pred_ids) if pred_ids else 0.0
        thr_r = tp / len(gold_ids) if gold_ids else 0.0
        thr_f1 = 2 * thr_p * thr_r / (thr_p + thr_r) if (thr_p + thr_r) > 0 else 0.0
        thresh_stats[t]["correct"]       += int(bool(tp))
        thresh_stats[t]["total"]         += 1
        thresh_stats[t]["precision_sum"] += thr_p
        thresh_stats[t]["recall_sum"]    += thr_r
        thresh_stats[t]["f1_sum"]        += thr_f1


def update_std_stats(std_stats, std_paths, gold_ids):
    """累加单样本标准束搜索（λ=0）指标。"""
    sm = compute_path_metrics(std_paths, gold_ids)
    std_stats["answer_path_hit"] += int(sm["answer_hit"])
    std_stats["top1_hit"]        += int(sm["top1_hit"])
    std_stats["precision_sum"]   += sm["precision"]
    std_stats["recall_sum"]      += sm["recall"]
    std_stats["f1_sum"]          += sm["f1"]
    std_stats["count"]           += 1


def print_validate_results(acc, hop_count, mmr_stats, thresh_stats, std_stats,
                            run_std, beam_size, lambda_val, acc_thresholds):
    """打印 qa_acc、hop acc、MMR 指标、e_score 阈值对比、标准束搜索 vs MMR 对比。"""
    c_mmr = mmr_stats["count"]

    print(acc)
    print('pred hop accuracy: 1-hop {} (total {}), 2-hop {} (total {})'.format(
        sum(hop_count[0]) / (len(hop_count[0]) + 0.1),
        len(hop_count[0]),
        sum(hop_count[1]) / (len(hop_count[1]) + 0.1),
        len(hop_count[1]),
    ))

    if c_mmr > 0:
        print('--- MMR Path Hit Metrics (beam_size={}, lambda={}) ---'.format(beam_size, lambda_val))
        print('qa_acc (original):      {:.4f}'.format(acc))
        print('mmr_answer_path_hit@{}: {:.4f}  ({}/{})'.format(
            beam_size, mmr_stats["answer_path_hit"] / c_mmr,
            mmr_stats["answer_path_hit"], c_mmr))
        print('mmr_precision@{}:       {:.4f}'.format(beam_size, mmr_stats["precision_sum"] / c_mmr))
        print('mmr_answer_recall@{}:   {:.4f}'.format(beam_size, mmr_stats["recall_sum"] / c_mmr))
        print('mmr_f1@{}:              {:.4f}'.format(beam_size, mmr_stats["f1_sum"] / c_mmr))
        print('mmr_top1_hit:           {:.4f}  ({}/{})'.format(
            mmr_stats["top1_hit"] / c_mmr, mmr_stats["top1_hit"], c_mmr))
        print('--- Path Diversity (avg over {} samples) ---'.format(c_mmr))
        print('jaccard_diversity:      {:.4f}  (0=完全重叠, 1=完全不同)'.format(
            mmr_stats["jaccard_div_sum"] / c_mmr))
        print('tail_diversity:         {:.4f}  (尾节点唯一率)'.format(
            mmr_stats["tail_div_sum"] / c_mmr))
        print('edge_coverage:          {:.4f}  (去重边数/总边数)'.format(
            mmr_stats["edge_cov_sum"] / c_mmr))

    # ── e_score 多阈值对比 ─────────────────────────────────────────
    print('\n--- e_score 阈值对比 (TransferNet 直接输出) ---')
    print('{:>8}  {:>8}  {:>10}  {:>8}  {:>8}'.format(
          '阈值', 'acc', 'precision', 'recall', 'f1'))
    for t in acc_thresholds:
        s = thresh_stats[t]
        n = s['total']
        print('  {:>6}  {:>8.4f}  {:>10.4f}  {:>8.4f}  {:>8.4f}'.format(
              t, s['correct'] / n, s['precision_sum'] / n,
              s['recall_sum'] / n, s['f1_sum'] / n))

    # ── 标准束搜索 vs MMR 对比 ────────────────────────────────────
    if run_std and std_stats['count'] > 0:
        c = std_stats['count']
        print('\n--- 标准束搜索 (λ=0) vs MMR (λ={}) 路径检索对比 ---'.format(lambda_val))
        print('{:>20}  {:>12}  {:>10}  {:>8}  {:>8}  {:>8}'.format(
              '方法', 'answer_hit', 'precision', 'recall', 'f1', 'top1'))
        print('  {:>18}  {:>12.4f}  {:>10.4f}  {:>8.4f}  {:>8.4f}  {:>8.4f}'.format(
              'MMR λ=' + str(lambda_val),
              mmr_stats["answer_path_hit"] / c_mmr, mmr_stats["precision_sum"] / c_mmr,
              mmr_stats["recall_sum"] / c_mmr, mmr_stats["f1_sum"] / c_mmr,
              mmr_stats["top1_hit"] / c_mmr))
        print('  {:>18}  {:>12.4f}  {:>10.4f}  {:>8.4f}  {:>8.4f}  {:>8.4f}'.format(
              '标准束搜索 λ=0',
              std_stats['answer_path_hit'] / c, std_stats['precision_sum'] / c,
              std_stats['recall_sum'] / c, std_stats['f1_sum'] / c,
              std_stats['top1_hit'] / c))
