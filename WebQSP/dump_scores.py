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
import datetime as _dt
import logging
import os
import sys

import torch
from tqdm import tqdm

from utils.misc import batch_device
from utils.path_utils import filter_tensor
from .predict import id_score_pairs
from .data import DataLoader, load_data
from .model import TransferNet


LOG_FORMAT = "%(asctime)s %(levelname)-8s %(message)s"
log = logging.getLogger(__name__)


def _default_log_path(output_path: str) -> str:
    output_abs = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_abs)
    stem = os.path.splitext(os.path.basename(output_abs))[0] or "webqsp_scores"
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_dir, f"{stem}_dump_{timestamp}.log")


def _setup_logging(log_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    for handler in list(root_logger.handlers):
        if getattr(handler, "_dump_scores_handler", False):
            root_logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(LOG_FORMAT)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler._dump_scores_handler = True

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler._dump_scores_handler = True

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)


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
    # 问句必须与得分同源：直接取 DataLoader 过滤后、与 batch 同序的问句文本，
    # 而非重新逐行读 QA 文件——后者不应用 DataLoader 的丢行过滤，会造成累积错位。
    raw_questions = getattr(data, "qa_text", None)
    if raw_questions is None:
        raise RuntimeError(
            "DataLoader 缺少 qa_text 属性，无法保证问句与得分对齐；请更新 WebQSP/data.py。"
        )
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

    if sample_counter != len(raw_questions):
        raise RuntimeError(
            f"问句数({len(raw_questions)})与得分样本数({sample_counter})不一致，"
            "二者必须严格同源同序；请检查 DataLoader 是否未关闭 shuffle 或过滤逻辑已变更。"
        )

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
                        choices=["bert-base-uncased", "BAAI/bge-base-en-v1.5", "roberta-base"])
    parser.add_argument("--output",     default="output/score_cache/webqsp_scores.pt",
                        help="缓存输出路径（.pt 文件）")
    parser.add_argument("--topk",       type=int, default=500,
                        help="每跳保存的实体得分 top-K 数量（默认 500）")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--qa_file",    required=True,
                        help="QA 文件路径（用于读取原始问题文本并构建 test loader）")
    parser.add_argument("--log_path",   default=None,
                        help="dump 日志输出路径；默认写入 --output 所在目录")
    args = parser.parse_args()

    log_path = args.log_path or _default_log_path(args.output)
    _setup_logging(log_path)
    log.info("dump log: %s", log_path)
    log.info("dump args: %s", vars(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("device=%s", device)

    log.info("加载数据 ...")
    ent2id, rel2id, triples, train_loader, val_loader = load_data(
        args.input_dir, args.bert_name, args.batch_size
    )
    if args.qa_file:
        qa_file = args.qa_file
        if not os.path.isabs(qa_file):
            qa_file = os.path.join(args.input_dir, qa_file)
        log.info("使用指定 QA 文件: %s", qa_file)
        val_loader = DataLoader(
            args.input_dir, qa_file, args.bert_name, ent2id, rel2id, args.batch_size
        )

    log.info("加载模型 ...")
    model = TransferNet(args, ent2id, rel2id, triples)
    missing, unexpected = model.load_state_dict(
        torch.load(args.ckpt, map_location="cpu"), strict=False
    )
    if missing:
        log.warning("Missing keys: %s", "; ".join(missing))
    if unexpected:
        log.warning("Unexpected keys: %s", "; ".join(unexpected))
    model = model.to(device)
    model.Msubj = model.Msubj.to(device)
    model.Mobj  = model.Mobj.to(device)
    model.Mrel  = model.Mrel.to(device)

    # dump 必须使用「按 --qa_file 构建、且未 shuffle」的 loader：
    # train_loader 是 training=True(shuffle=True)，会打乱 batch 顺序导致问句/得分错位；
    # 且 --qa_file 为 required，上面的 override 已把 val_loader 重建为该非 shuffle loader。
    # mode 仅作为 cache 元数据标签使用。
    loader = val_loader
    dump_scores(
        model, loader, device, args.output,
        topk=args.topk,
        mode=args.mode,
        input_dir=args.input_dir,
        qa_file=args.qa_file,
    )
    log.info("dump log written: %s", log_path)


if __name__ == "__main__":
    main()
