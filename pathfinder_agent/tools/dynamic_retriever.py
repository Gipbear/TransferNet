# pathfinder_agent/tools/dynamic_retriever.py
import json
import os

from pathfinder_agent.config import BEAM_SIZE_PRIMARY, LAMBDA_PRIMARY, BEAM_SIZE_FALLBACK, LAMBDA_FALLBACK, MAX_PATHS_RETURNED, MAX_DUPLICATE_TAIL_NODES


def _load_entity_name_map(data_dir: str) -> dict:
    """从 data_dir 下合并加载 entities_names.json 与 mapped_entities.txt。

    两个来源互补：
      - entities_names.json: JSON 格式，键为 MID，值为名称
      - fbwq_full/mapped_entities.txt: TSV 格式，per-line "MID\tName"

    优先级：entities_names.json > mapped_entities.txt（前者更精确）。
    找不到文件时静默跳过，返回空 dict（不影响 MID 格式路径的正常工作）。
    """
    name_map: dict = {}

    # 1. mapped_entities.txt（覆盖面更广）
    for candidate in [
        os.path.join(data_dir, "mapped_entities.txt"),
        os.path.join(data_dir, "fbwq_full", "mapped_entities.txt"),
        os.path.join(os.path.dirname(data_dir), "fbwq_full", "mapped_entities.txt"),
    ]:
        if os.path.exists(candidate):
            with open(candidate, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t", 1)
                    if len(parts) == 2:
                        name_map[parts[0]] = parts[1]
            break

    # 2. entities_names.json（优先级更高，覆盖已有条目）
    for candidate in [
        os.path.join(data_dir, "entities_names.json"),
        os.path.join(data_dir, "fbwq_full", "entities_names.json"),
        os.path.join(os.path.dirname(data_dir), "fbwq_full", "entities_names.json"),
    ]:
        if os.path.exists(candidate):
            with open(candidate, encoding="utf-8") as f:
                name_map.update(json.load(f))
            break

    return name_map


def _resolve_path_names(paths: list, entity_name_map: dict) -> list:
    """将路径中的实体 MID 替换为人类可读名称（best-effort）。

    设计原则：
      - 有名称的实体 → 替换为名称（帮助 LLM 理解语义、识别噪声路径）
      - CVT 节点 / 无名称 MID → 保留原始 MID
        ∵ CVT 节点天然无名称，保留 MID 形式可让 LLM 将其识别为中间占位节点

    返回新列表，不修改原始 paths（每条路径新增 "mid_path" 字段保存原始 MID 列表）。
    """
    if not entity_name_map:
        return paths

    resolved = []
    for p in paths:
        mid_edges = p.get("path", [])
        name_edges = [
            [
                entity_name_map.get(e[0], e[0]),
                e[1],
                entity_name_map.get(e[2], e[2]),
            ]
            for e in mid_edges
        ]
        resolved.append({
            "path": name_edges,       # LLM 看到名称版本
            "mid_path": mid_edges,    # 保留原始 MID，供评测/调试使用
            "log_score": p.get("log_score", 0.0),
        })
    return resolved


def _needs_high_recall_fallback(question: str) -> bool:
    lowered = f" {question.lower()} "
    return any(
        token in lowered
        for token in (
            " first ",
            " second ",
            " third ",
            " when ",
            " before ",
            " after ",
            " during ",
            " 19",
            " 20",
            " governor ",
            " vice president ",
            " representatives ",
            " team ",
            " play for",
            " played for",
        )
    )


def get_retrieval_policy(question: str, fallback: bool = False) -> dict:
    high_recall = _needs_high_recall_fallback(question)
    if fallback:
        return {
            "beam_size": BEAM_SIZE_FALLBACK,
            "lambda_val": LAMBDA_FALLBACK,
            "max_paths_returned": MAX_PATHS_RETURNED + (10 if high_recall else 0),
            "max_duplicate_tail_nodes": MAX_DUPLICATE_TAIL_NODES + (2 if high_recall else 0),
        }
    return {
        "beam_size": BEAM_SIZE_PRIMARY,
        "lambda_val": LAMBDA_PRIMARY,
        "max_paths_returned": MAX_PATHS_RETURNED,
        "max_duplicate_tail_nodes": MAX_DUPLICATE_TAIL_NODES,
    }


class TransferNetWrapper:
    def __init__(self, data_dir, ckpt_path):
        """
        初始化 TransferNet，载入字典和模型。
        """
        import torch
        from WebQSP.model import TransferNet
        from WebQSP.data import load_data
        from utils.path_utils import build_valid_edges_dict
        import argparse
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        args = argparse.Namespace(
            input_dir=data_dir,
            ckpt=ckpt_path,
            bert_name='bert-base-uncased'
        )
        
        self.ent2id, self.rel2id, self.triples, _, _ = load_data(args.input_dir, args.bert_name, batch_size=1)
        self.model = TransferNet(args, self.ent2id, self.rel2id, self.triples)
        self.model.load_state_dict(torch.load(args.ckpt), strict=False)
        self.model.to(self.device)
        self.model.Msubj = self.model.Msubj.to(self.device)
        self.model.Mobj  = self.model.Mobj.to(self.device)
        self.model.Mrel  = self.model.Mrel.to(self.device)
        self.model.eval()
        
        triples_list = [[int(s), int(r), int(o)] for s, r, o in self.triples.tolist()]
        self.valid_edges_dict = build_valid_edges_dict(triples_list)

        from utils.misc import invert_dict
        self.id2ent = invert_dict(self.ent2id)
        self.id2rel = invert_dict(self.rel2id)

        # 加载实体名称映射（best-effort，找不到文件时返回空 dict）
        self.entity_name_map = _load_entity_name_map(data_dir)

    def retrieve(self, question: str, topic_entity: str, fallback=False):
        """
        Run TransferNet inference for a single question and return MMR paths.

        Args:
            question:      Natural language question string.
            topic_entity:  Topic entity name (must exist in ent2id).
            fallback:      If True, use BEAM_SIZE_FALLBACK / LAMBDA_FALLBACK.

        Returns:
            List of dicts: [{"path": [[head, rel, tail], ...], "log_score": float}, ...]
            after tail-node dedup and size capping.
        """
        import torch
        from utils.path_utils import mmr_diversity_beam_search, filter_tensor

        policy = get_retrieval_policy(question, fallback=fallback)
        beam_size = policy["beam_size"]
        lam = policy["lambda_val"]

        # -- Build topic-entity one-hot vector --
        if topic_entity not in self.ent2id:
            # Gracefully degrade: return empty
            return []
        eid = self.ent2id[topic_entity]
        topic_vec = torch.zeros(len(self.ent2id))
        topic_vec[eid] = 1.0
        topic_vec = topic_vec.to(self.device)

        # -- Tokenize question using BERT tokenizer --
        from transformers import AutoTokenizer
        if not hasattr(self, "_tokenizer"):
            self._tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        q_enc = self._tokenizer(
            question.strip(), max_length=64,
            padding="max_length", return_tensors="pt"
        )
        q_enc = {k: v.to(self.device) for k, v in q_enc.items()}

        # -- Reshape for batch_size=1 --
        topic_vec = topic_vec.unsqueeze(0)  # [1, E]

        with torch.no_grad():
            outputs = self.model(topic_vec, q_enc, None, None)

        hop_attn = outputs["hop_attn"][0]
        h = hop_attn.argmax().item()

        single_outputs = {
            "rel_probs": [outputs["rel_probs"][t][0].cpu() for t in range(len(outputs["rel_probs"]))],
            "ent_probs": [outputs["ent_probs"][t][0].cpu() for t in range(len(outputs["ent_probs"]))],
        }

        # Precompute filtered dicts for speed
        precomputed = [
            (dict(filter_tensor(single_outputs["rel_probs"][t], 0.01)),
             dict(filter_tensor(single_outputs["ent_probs"][t], 0.01)))
            for t in range(h + 1)
        ]

        topic_scores = [(eid, 1.0)]  # (entity_id, score)

        mmr_paths = mmr_diversity_beam_search(
            single_outputs, self.valid_edges_dict, topic_scores,
            h + 1, K=beam_size, lambda_val=lam,
            precomputed_dicts=precomputed,
        )

        # Serialize to dict format consistent with predict.py output
        reason_paths = []
        for nodes, rels, score in mmr_paths:
            reason_paths.append({
                "path": [
                    [self.id2ent.get(nodes[k], str(nodes[k])),
                     self.id2rel.get(rels[k],  str(rels[k])),
                     self.id2ent.get(nodes[k + 1], str(nodes[k + 1]))]
                    for k in range(len(rels))
                ],
                "log_score": round(float(score), 6),
            })

        # Apply tail-node dedup + size cap
        filtered = _filter_by_tail_node(reason_paths, policy["max_duplicate_tail_nodes"])
        filtered = filtered[:policy["max_paths_returned"]]

        # 将实体 MID 替换为人类可读名称（CVT 节点无名称，保留 MID 作为自然标识）
        return _resolve_path_names(filtered, self.entity_name_map)

def retrieve_paths(wrapper: TransferNetWrapper, question: str, topic_entity: str, fallback=False):
    """
    根据给定的问题执行实体路径检索。
    """
    if wrapper is None:
        return []
    
    return wrapper.retrieve(question, topic_entity, fallback=fallback)
    
def _filter_by_tail_node(paths, max_duplicates):
    """
    根据路径到达的尾节点去重，使得相同尾节点的候选路径不超过 max_duplicates 条。
    paths: list of dict, e.g. [{"path": [["E1", "R1", "E2"]], "log_score": 0.8}, ...]
    按照得分从高到低排序，同一个实体后缀最多保留2条。
    """
    # 先按照分数降序排序
    sorted_paths = sorted(paths, key=lambda x: x.get('log_score', 0.0), reverse=True)
    
    tail_counts = {}
    filtered_paths = []
    
    for p in sorted_paths:
        path_list = p.get('path', [])
        if not path_list:
            continue
        # path_list 形式为 [ [head, rel, tail], [head, rel, tail] ]
        # 获取整条路径的终点
        tail_node = path_list[-1][-1]
        
        count = tail_counts.get(tail_node, 0)
        if count < max_duplicates:
            filtered_paths.append(p)
            tail_counts[tail_node] = count + 1
            
    return filtered_paths
