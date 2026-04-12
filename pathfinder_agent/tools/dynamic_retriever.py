# pathfinder_agent/tools/dynamic_retriever.py
from pathfinder_agent.config import BEAM_SIZE_PRIMARY, LAMBDA_PRIMARY, BEAM_SIZE_FALLBACK, LAMBDA_FALLBACK, MAX_PATHS_RETURNED, MAX_DUPLICATE_TAIL_NODES

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

        beam_size = BEAM_SIZE_FALLBACK if fallback else BEAM_SIZE_PRIMARY
        lam       = LAMBDA_FALLBACK   if fallback else LAMBDA_PRIMARY

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
        filtered = _filter_by_tail_node(reason_paths, MAX_DUPLICATE_TAIL_NODES)
        return filtered[:MAX_PATHS_RETURNED]

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
