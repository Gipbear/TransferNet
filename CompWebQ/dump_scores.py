"""CWQ 得分矩阵导出脚本。

运行 CompWebQ TransferNet 推理，将每个样本的中间得分矩阵和逐样本
subgraph triples 写入 .pt 缓存，供 scripts/offline_path_search.py 离线重放。
"""
from __future__ import annotations

import argparse
import os

import torch
from tqdm import tqdm

from utils.misc import batch_device
from .data import load_data
from .model import TransferNet


def _decode_question(tokenizer, question_batch: dict[str, torch.Tensor], row: int) -> str:
    input_ids = question_batch["input_ids"][row].detach().cpu().tolist()
    return tokenizer.decode(input_ids, skip_special_tokens=True).strip()


def dump_scores(model, data, device, output_path, topk=500, mode="val", input_dir=None):
    """运行 CWQ 推理，将中间得分和样本 subgraph 保存为离线缓存。"""
    model.eval()
    samples = []
    sample_counter = 0

    pbar = tqdm(data, total=len(data), desc="dump_scores", unit="batch", dynamic_ncols=True)
    with torch.no_grad():
        for batch in pbar:
            outputs = model(*batch_device(batch, device))

            e_score_cpu = outputs["e_score"].cpu()
            hop_attn_cpu = outputs["hop_attn"].cpu()
            rel_probs_cpu = [t.cpu() for t in outputs["rel_probs"]]
            ent_probs_cpu = [t.cpu() for t in outputs["ent_probs"]]
            num_steps = len(rel_probs_cpu)

            bsz = e_score_cpu.shape[0]
            for i in range(bsz):
                topic_ids = batch[0][i].tolist()
                gold_ids = batch[2][i].tolist()
                triples = batch[3][i].detach().cpu().long().tolist()
                question = _decode_question(data.tokenizer, batch[1], i)
                sample_counter += 1

                ent_indices_per_hop, ent_scores_per_hop = [], []
                for t in range(num_steps):
                    vec = ent_probs_cpu[t][i]
                    k = min(topk, vec.shape[0])
                    top_vals, top_idxs = vec.topk(k)
                    mask = top_vals > 0
                    ent_indices_per_hop.append(top_idxs[mask])
                    ent_scores_per_hop.append(top_vals[mask])

                e_vec = e_score_cpu[i]
                k = min(topk, e_vec.shape[0])
                e_top_vals, e_top_idxs = e_vec.topk(k)
                e_mask = e_top_vals > 0

                samples.append({
                    "question": question,
                    "topic_ids": topic_ids,
                    "gold_ids": gold_ids,
                    "triples": triples,
                    "hop_attn": hop_attn_cpu[i].clone(),
                    "rel_probs": [rel_probs_cpu[t][i].clone() for t in range(num_steps)],
                    "ent_indices": ent_indices_per_hop,
                    "ent_scores": ent_scores_per_hop,
                    "e_score_indices": e_top_idxs[e_mask],
                    "e_score_values": e_top_vals[e_mask],
                })

            pbar.set_postfix(samples=sample_counter)
            del outputs, e_score_cpu, hop_attn_cpu, rel_probs_cpu, ent_probs_cpu

    cache = {
        "version": 1,
        "meta": {
            "dataset": "CWQ",
            "split": mode,
            "num_samples": len(samples),
            "num_entities": len(data.id2ent),
            "num_relations": len(data.id2rel),
            "num_steps": model.num_steps,
            "topk_entities": topk,
            "input_dir": input_dir,
            "graph_source": "sample_triples",
            "id2ent": data.id2ent,
            "id2rel": data.id2rel,
        },
        "samples": samples,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(cache, output_path)
    print(f"[INFO] 得分缓存已写入: {output_path}  ({len(samples)} 条样本)", flush=True)
    return cache


def main():
    parser = argparse.ArgumentParser(description="CWQ TransferNet 得分矩阵导出")
    parser.add_argument("--input_dir", required=True, help="CWQ 数据目录")
    parser.add_argument("--ckpt", required=True, help="模型 checkpoint 路径")
    parser.add_argument("--mode", default="val", choices=["train", "val", "dev", "test"],
                        help="使用哪个数据集分割（默认: val/dev）")
    parser.add_argument("--bert_name", default="BAAI/bge-base-en-v1.5",
                        choices=["roberta-base", "bert-base-cased", "bert-base-uncased",
                                 "BAAI/bge-base-en-v1.5"])
    parser.add_argument("--output", default="output/score_cache/cwq_scores.pt",
                        help="缓存输出路径（.pt 文件）")
    parser.add_argument("--topk", type=int, default=500,
                        help="每跳保存的实体得分 top-K 数量（默认 500）")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=2)
    parser.add_argument("--num_ways", type=int, default=1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}", flush=True)

    print("[INFO] 加载数据 ...", flush=True)
    ent2id, rel2id, train_loader, dev_loader, test_loader = load_data(
        args.input_dir, args.bert_name, args.batch_size
    )

    print("[INFO] 加载模型 ...", flush=True)
    model = TransferNet(args, ent2id, rel2id)
    missing, unexpected = model.load_state_dict(
        torch.load(args.ckpt, map_location="cpu"), strict=False
    )
    if missing:
        print("Missing keys: {}".format("; ".join(missing)))
    if unexpected:
        print("Unexpected keys: {}".format("; ".join(unexpected)))
    model = model.to(device)

    if args.mode == "train":
        loader = train_loader
    elif args.mode == "test":
        loader = test_loader
    else:
        loader = dev_loader

    dump_scores(
        model, loader, device, args.output,
        topk=args.topk,
        mode=args.mode,
        input_dir=args.input_dir,
    )


if __name__ == "__main__":
    main()
