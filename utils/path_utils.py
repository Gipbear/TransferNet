"""
共享路径工具函数，供 WebQSP / CompWebQ / MetaQA_KB 等数据集共同复用。
不依赖任何数据集特定逻辑，只依赖 torch 和标准库。

包含：
  filter_tensor            - 张量阈值过滤
  path_to_edge_set         - 路径转有向边三元组集合
  path_to_rel_set          - 路径转带 hop 位置的关系集合
  compute_path_metrics     - 路径命中率/精度/召回/F1
  compute_path_diversity   - 路径多样性（边级/关系级 Jaccard、尾节点、关系覆盖）
  mmr_diversity_beam_search - 关系级 MMR 多样性束搜索（WebQSP/MetaQA_KB 共用）
  build_valid_edges_dict   - 三元组列表转边查找字典
"""
import math
import torch
from collections import defaultdict

EPS = 1e-9


def filter_tensor(tensor, threshold=0.9):
    indices = torch.where(tensor >= threshold)[0]
    scores = tensor[indices]
    return list(zip(indices.tolist(), scores.tolist()))


def path_to_edge_set(nodes, rels):
    """将路径的节点列表和关系列表转为有向边三元组集合，用于边级 Jaccard 相似度计算。"""
    return set(zip(nodes[:-1], rels, nodes[1:]))


def path_to_rel_set(rels):
    """将路径的关系列表转为 (hop_idx, rel) 集合，用于关系级 Jaccard 相似度计算。
    同一关系不同尾节点的路径相似度为 1.0，防止同关系路径霸占 beam。"""
    return set(enumerate(rels))


def compute_path_metrics(paths, gold_ids):
    """计算路径检索指标：answer_hit / top1_hit / precision / recall / f1。"""
    tail_ids   = {nodes[-1] for nodes, _, _ in paths if len(nodes) > 1}
    hit_count  = len(tail_ids & gold_ids)
    answer_hit = bool(hit_count)
    top1_hit   = bool(paths and paths[0][0][-1] in gold_ids)
    recall     = hit_count / len(gold_ids) if gold_ids else 0.0
    precision  = hit_count / len(tail_ids) if tail_ids else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {"answer_hit": answer_hit, "top1_hit": top1_hit,
            "recall": recall, "precision": precision, "f1": f1}


def compute_path_diversity(paths):
    """
    计算 K 条路径的多样性指标。
      jaccard_diversity          : 边级平均成对 Jaccard 距离（1 - 相似度），范围 [0, 1]
      relation_jaccard_diversity : 关系级平均成对 Jaccard 距离（1 - 相似度），范围 [0, 1]
      tail_diversity             : 尾节点唯一率 = |unique tails| / K，范围 [1/K, 1]
      relation_coverage          : 去重关系数 / 总关系数，越高说明路径覆盖了更多不同的关系模式
    paths: list of (nodes, rels, score)
    """
    if len(paths) < 2:
        return {
            "jaccard_diversity": 0.0,
            "relation_jaccard_diversity": 0.0,
            "tail_diversity": 0.0,
            "relation_coverage": 0.0,
            "edge_coverage": 0.0,
        }

    edge_sets = [path_to_edge_set(nodes, rels) for nodes, rels, _ in paths]
    rel_sets = [path_to_rel_set(rels) for _, rels, _ in paths]
    K = len(edge_sets)

    # 1. 平均成对边级 Jaccard 距离
    edge_pair_sims = []
    rel_pair_sims = []
    for i in range(K):
        for j in range(i + 1, K):
            edge_union = edge_sets[i] | edge_sets[j]
            edge_sim = len(edge_sets[i] & edge_sets[j]) / len(edge_union) if edge_union else 0.0
            edge_pair_sims.append(edge_sim)

            rel_union = rel_sets[i] | rel_sets[j]
            rel_sim = len(rel_sets[i] & rel_sets[j]) / len(rel_union) if rel_union else 0.0
            rel_pair_sims.append(rel_sim)
    jaccard_diversity = 1.0 - (sum(edge_pair_sims) / len(edge_pair_sims))
    relation_jaccard_diversity = 1.0 - (sum(rel_pair_sims) / len(rel_pair_sims))

    # 2. 尾节点唯一率
    tails = [nodes[-1] for nodes, _, _ in paths if len(nodes) > 1]
    tail_diversity = len(set(tails)) / K if tails else 0.0

    # 3. 关系覆盖率
    all_relations = set().union(*rel_sets)
    total_relations = sum(len(r) for r in rel_sets)
    relation_coverage = len(all_relations) / total_relations if total_relations > 0 else 0.0

    # 4. 边集覆盖率
    all_edges = set().union(*edge_sets)
    total_edges = sum(len(e) for e in edge_sets)
    edge_coverage = len(all_edges) / total_edges if total_edges > 0 else 0.0

    return {
        "jaccard_diversity": round(jaccard_diversity, 4),
        "relation_jaccard_diversity": round(relation_jaccard_diversity, 4),
        "tail_diversity":    round(tail_diversity, 4),
        "relation_coverage": round(relation_coverage, 4),
        "edge_coverage":     round(edge_coverage, 4),
    }


def mmr_diversity_beam_search(outputs, valid_edges_dict, start_entities, hop_num,
                               K=3, lambda_val=0.5, precomputed_dicts=None):
    """
    基于 MMR 的多跳路径广度优先束搜索（WebQSP / MetaQA_KB 共用）。
    :param outputs: 包含 'rel_probs' 和 'ent_probs' 的 dict，元素为 CPU 1-D tensor
    :param valid_edges_dict: 预先构建好的 {subj: [(rel, obj), ...]} 边查找字典
    :param start_entities: (entity_id, score) 元组列表
    :param hop_num: 跳数
    :param precomputed_dicts: 可选，list of (rel_dict, ent_dict)，按 hop 索引，
                              用于多次调用共享同一份 filter_tensor 结果，避免重复计算
    """
    beam = [([ent], [], 0.0) for ent, _ in start_entities]

    for t in range(hop_num):
        candidates = []
        if precomputed_dicts is not None:
            rel_dict, ent_dict = precomputed_dicts[t]
        else:
            rel_dict = dict(filter_tensor(outputs['rel_probs'][t], 0.01))
            ent_dict = dict(filter_tensor(outputs['ent_probs'][t], 0.01))

        # 方案 A: 预计算全局实体分数总和，用于全局归一化（消除高扇出关系的惩罚）
        global_ent_sum = sum(ent_dict.values()) + EPS

        for nodes, rels, current_score in beam:
            u = nodes[-1]
            possible_out_edges = valid_edges_dict.get(u, [])

            # 收集局部关系分数和有效边列表
            local_rel_scores: dict = {}
            filtered_edges = []
            for (r, v) in possible_out_edges:
                if r in rel_dict:
                    if r not in local_rel_scores:
                        local_rel_scores[r] = rel_dict[r]
                    if v in ent_dict:
                        filtered_edges.append((r, v))
            local_rel_sum = sum(local_rel_scores.values()) + EPS

            for (r_valid, v_valid) in filtered_edges:
                local_rel_prob = local_rel_scores[r_valid] / local_rel_sum
                # 方案 A: 使用全局归一化替代关系内归一化，消除高扇出关系的惩罚
                local_ent_prob = ent_dict[v_valid] / global_ent_sum
                step_score = math.log(local_rel_prob + EPS) + math.log(local_ent_prob + EPS)
                candidates.append((nodes + [v_valid], rels + [r_valid], current_score + step_score))

        if not candidates:
            break

        # 方案 C: MMR 筛选（关系级 Jaccard 相似度）
        # 同关系不同尾节点的路径相似度 = 1.0，防止同关系路径霸占 beam
        cand_rel_sets = [path_to_rel_set(c[1]) for c in candidates]
        next_beam = []
        next_beam_rels = []           # 缓存已选路径的 rel_set
        remaining_idx = list(range(len(candidates)))
        while len(next_beam) < K and remaining_idx:
            best_idx, best_mmr = None, -float('inf')
            for idx in remaining_idx:
                cand_score = candidates[idx][2]
                cand_rels = cand_rel_sets[idx]
                if next_beam_rels:
                    max_sim = max(
                        len(cand_rels & br) / len(cand_rels | br)
                        if (cand_rels | br) else 0.0
                        for br in next_beam_rels
                    )
                else:
                    max_sim = 0.0
                mmr_score = cand_score - lambda_val * max_sim * abs(cand_score)
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx
            next_beam.append(candidates[best_idx])
            next_beam_rels.append(cand_rel_sets[best_idx])
            remaining_idx.remove(best_idx)

        beam = next_beam

    return beam  # list of (nodes, rels, log_score)


def build_valid_edges_dict(triples_list):
    """从三元组列表构建 {subj: [(rel, obj), ...]} 查找字典。"""
    d = defaultdict(list)
    for s, r, o in triples_list:
        d[s].append((r, o))
    return d
