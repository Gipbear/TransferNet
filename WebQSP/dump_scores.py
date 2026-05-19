"""WebQSP 得分矩阵导出脚本

独立运行 TransferNet 推理，将每个样本的中间得分矩阵（每跳的实体/关系得分、
跳数权重）保存到 .pt 缓存文件，供后续离线路径搜索实验使用。

不修改 predict.py，不与原有推理/路径搜索逻辑耦合。

用法：
  python -m WebQSP.dump_scores \\
      --ckpt data/ckpt/WebQSP/model.pt \\
      --input_dir data/WebQSP \\
      --mode val \\
      --output output/score_cache/webqsp_val.pt \\
      --topk 500
"""
import argparse
import os

import torch
from tqdm import tqdm

from utils.misc import batch_device
from utils.path_utils import filter_tensor
from .predict import id_score_pairs
from .data import DataLoader, load_data
from .model import TransferNet


def _load_raw_questions(qa_file: str) -> list[str]:
    """从 WebQSP QA 文件按顺序提取原始问题文本（去掉末尾的 [topic_mid] 和答案列）。"""
    questions = []
    with open(qa_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q_part = line.split("\t")[0]
            if " [" in q_part and q_part.endswith("]"):
                q_part = q_part.rsplit(" [", 1)[0]
            questions.append(q_part.strip())
    return questions


def dump_scores(model, data, device, output_path, topk=500, mode="val",
                input_dir=None, qa_file=None):
    """运行推理，将每个样本的中间得分矩阵写入 .pt 缓存文件。

    缓存格式：
    {
        "version": 1,
        "meta": {
            "dataset": "WebQSP",
            "split": str,          # val / test / train
            "num_samples": int,
            "num_entities": int,
            "num_relations": int,
            "num_steps": int,      # 模型跳数（固定为 2）
            "topk_entities": int,  # 每跳保存的实体 top-K
            "id2ent": dict,        # int -> MID str
            "id2rel": dict,        # int -> rel str
        },
        "samples": [
            {
                "question": str,           # 原始问题文本（从 QA 文件读取，保留原始标点）
                "topic_ids": list[int],    # topic 实体 ID 列表
                "gold_ids":  list[int],    # gold 答案实体 ID 列表
                "hop_attn":  Tensor[num_steps],      # 模型对各跳的注意力权重
                "rel_probs": list[Tensor[num_rel]],  # 每跳关系得分（密集，sigmoid 输出）
                "ent_indices": list[Tensor[K']],     # 每跳实体 top-K 索引（稀疏）
                "ent_scores":  list[Tensor[K']],     # 每跳实体 top-K 得分（稀疏）
                "e_score_indices": Tensor[K'],       # 最终聚合实体得分 top-K 索引
                "e_score_values":  Tensor[K'],       # 最终聚合实体得分 top-K 值
            },
            ...
        ]
    }

    实体得分以稀疏 top-K 存储（原始 Esize ~45K，实际有效通常 <200）。
    关系得分以密集向量存储（num_relations ~700，存储代价极小）。
    """
    model.eval()
    samples = []

    if not qa_file or not os.path.isfile(qa_file):
        raise ValueError(f"qa_file 必须提供且存在，当前值: {qa_file!r}")
    raw_questions = _load_raw_questions(qa_file)
    sample_counter = 0

    pbar = tqdm(data, total=len(data), desc="dump_scores", unit="batch", dynamic_ncols=True)
    with torch.no_grad():
        for batch in pbar:
            outputs = model(*batch_device(batch, device))

            e_score_cpu   = outputs['e_score'].cpu()                        # [bsz, Esize]
            hop_attn_cpu  = outputs['hop_attn'].cpu()                       # [bsz, num_steps]
            rel_probs_cpu = [t.cpu() for t in outputs['rel_probs']]         # list of [bsz, num_rel]
            ent_probs_cpu = [t.cpu() for t in outputs['ent_probs']]         # list of [bsz, Esize]
            num_steps = len(rel_probs_cpu)

            bsz = e_score_cpu.shape[0]
            for i in range(bsz):
                topic_ids = [x for (x, _) in id_score_pairs(batch[0][i], 1)]
                gold_ids  = [x for (x, _) in id_score_pairs(batch[2][i], 1)]

                question = raw_questions[sample_counter]
                sample_counter += 1

                ent_indices_per_hop, ent_scores_per_hop = [], []
                for t in range(num_steps):
                    vec = ent_probs_cpu[t][i]
                    k   = min(topk, vec.shape[0])
                    top_vals, top_idxs = vec.topk(k)
                    mask = top_vals > 0
                    ent_indices_per_hop.append(top_idxs[mask])
                    ent_scores_per_hop.append(top_vals[mask])

                e_vec = e_score_cpu[i]
                k     = min(topk, e_vec.shape[0])
                e_top_vals, e_top_idxs = e_vec.topk(k)
                e_mask = e_top_vals > 0

                samples.append({
                    "question":        question,
                    "topic_ids":       topic_ids,
                    "gold_ids":        gold_ids,
                    "hop_attn":        hop_attn_cpu[i].clone(),
                    "rel_probs":       [rel_probs_cpu[t][i].clone() for t in range(num_steps)],
                    "ent_indices":     ent_indices_per_hop,
                    "ent_scores":      ent_scores_per_hop,
                    "e_score_indices": e_top_idxs[e_mask],
                    "e_score_values":  e_top_vals[e_mask],
                })

            pbar.set_postfix(samples=sample_counter)
            del outputs, e_score_cpu, hop_attn_cpu, rel_probs_cpu, ent_probs_cpu

    cache = {
        "version": 1,
        "meta": {
            "dataset":       "WebQSP",
            "split":         mode,
            "num_samples":   len(samples),
            "num_entities":  len(data.id2ent),
            "num_relations": len(data.id2rel),
            "num_steps":     model.num_steps,
            "topk_entities": topk,
            "input_dir":     input_dir,
            "qa_file":       qa_file,
            "id2ent":        data.id2ent,
            "id2rel":        data.id2rel,
        },
        "samples": samples,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(cache, output_path)
    print(f"[INFO] 得分缓存已写入: {output_path}  ({len(samples)} 条样本)", flush=True)
    return cache


def main():
    parser = argparse.ArgumentParser(description="WebQSP TransferNet 得分矩阵导出")
    parser.add_argument("--input_dir",  required=True,
                        help="WebQSP 数据目录（含 fbwq_full/）")
    parser.add_argument("--ckpt",       required=True,
                        help="模型 checkpoint 路径")
    parser.add_argument("--mode",       default="val",
                        choices=["val", "test", "train"],
                        help="使用哪个数据集分割（默认: val）")
    parser.add_argument("--bert_name",  default="bert-base-uncased",
                        choices=["bert-base-uncased", "roberta-base"])
    parser.add_argument("--output",     default="output/score_cache/webqsp_scores.pt",
                        help="缓存输出路径（.pt 文件）")
    parser.add_argument("--topk",       type=int, default=500,
                        help="每跳保存的实体得分 top-K 数量（默认 500）")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--qa_file",    required=True,
                        help="QA 文件路径（用于读取原始问题文本并构建 test loader）")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}", flush=True)

    print("[INFO] 加载数据 ...", flush=True)
    ent2id, rel2id, triples, train_loader, val_loader = load_data(
        args.input_dir, args.bert_name, args.batch_size
    )
    if args.qa_file:
        qa_file = args.qa_file
        if not os.path.isabs(qa_file):
            qa_file = os.path.join(args.input_dir, qa_file)
        print(f"[INFO] 使用指定 QA 文件: {qa_file}", flush=True)
        val_loader = DataLoader(
            args.input_dir, qa_file, args.bert_name, ent2id, rel2id, args.batch_size
        )

    print("[INFO] 加载模型 ...", flush=True)
    model = TransferNet(args, ent2id, rel2id, triples)
    missing, unexpected = model.load_state_dict(
        torch.load(args.ckpt, map_location="cpu"), strict=False
    )
    if missing:
        print("Missing keys: {}".format("; ".join(missing)))
    if unexpected:
        print("Unexpected keys: {}".format("; ".join(unexpected)))
    model = model.to(device)
    model.Msubj = model.Msubj.to(device)
    model.Mobj  = model.Mobj.to(device)
    model.Mrel  = model.Mrel.to(device)

    loader = train_loader if args.mode == "train" else val_loader
    dump_scores(
        model, loader, device, args.output,
        topk=args.topk,
        mode=args.mode,
        input_dir=args.input_dir,
        qa_file=args.qa_file,
    )


if __name__ == "__main__":
    main()
