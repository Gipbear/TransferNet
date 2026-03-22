import json
import os
import torch
import argparse
from tqdm import tqdm
from collections import defaultdict
from utils.misc import batch_device
from utils.path_utils import filter_tensor
from .data import load_data
from .model import TransferNet
from utils.path_utils import mmr_diversity_beam_search, build_valid_edges_dict
from utils.eval_utils import (
    create_mmr_stats, create_thresh_stats, create_std_stats,
    update_mmr_stats, update_thresh_stats, update_std_stats,
    print_validate_results,
)

from IPython import embed


def validate(args, model, data, triples_list, valid_edges_dict, device,
             verbose=False, beam_size=3, lambda_val=0.5, output_path=None,
             mid2label=None, acc_thresholds=None, compare_standard=True):
    if acc_thresholds is None:
        acc_thresholds = [0.7, 0.8, 0.9]

    thresh_stats = create_thresh_stats(acc_thresholds)

    # 标准束搜索对比（仅当 lambda_val != 0 时才有对比意义）
    run_std = compare_standard and (lambda_val != 0.0)
    std_stats = create_std_stats()

    model.eval()
    count = 0
    correct = 0
    hop_count = defaultdict(list)
    mmr_stats = create_mmr_stats()

    out_path = output_path or "WebQSP/predict_result_webqsp_path_info.jsonl"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    open(out_path, 'w').close()  # 清空旧内容

    # MID → 可读名的辅助函数，无映射时原样返回
    def resolve(mid: str) -> str:
        if mid2label:
            return mid2label.get(mid, mid)
        return mid

    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            outputs = model(*batch_device(batch, device))  # [bsz, Esize]
            e_score = outputs['e_score'].cpu()
            scores, idx = torch.max(e_score, dim=1)  # [bsz], [bsz]
            pred_mask = e_score.gt(0.7)
            hit_score = (pred_mask & batch[2].gt(0.7)).any(dim=1).float().tolist()
            del pred_mask
            count += len(hit_score)
            correct += sum(hit_score)
            hop_attn_cpu  = outputs['hop_attn'].cpu()
            # 批量转移到 CPU，避免在 i 循环内逐 sample 切片转移造成内存峰值叠加
            rel_probs_cpu = [t.cpu() for t in outputs['rel_probs']]
            ent_probs_cpu = [t.cpu() for t in outputs['ent_probs']]

            batch_infos = []
            for i in range(len(hit_score)):
                h = hop_attn_cpu[i].argmax().item()
                hop_count[h].append(hit_score[i])

                topic_scores = filter_tensor(batch[0][i], 1)
                topics = [resolve(data.id2ent[x]) for (x, _) in topic_scores]

                # 一次性计算 gold 信息，后续（std/mmr 指标、输出序列化）三处复用
                gold_pairs = filter_tensor(batch[2][i], 1)
                gold_ids   = {x for (x, _) in gold_pairs}

                single_outputs = {
                    'rel_probs': [rel_probs_cpu[t][i] for t in range(len(rel_probs_cpu))],
                    'ent_probs': [ent_probs_cpu[t][i] for t in range(len(ent_probs_cpu))],
                }

                # 预计算 rel_dict / ent_dict，MMR 和标准束搜索共享，避免重复 filter_tensor
                precomputed = [
                    (dict(filter_tensor(single_outputs['rel_probs'][t], 0.01)),
                     dict(filter_tensor(single_outputs['ent_probs'][t], 0.01)))
                    for t in range(h + 1)
                ]

                mmr_paths = mmr_diversity_beam_search(
                    single_outputs, valid_edges_dict, topic_scores,
                    h + 1, K=beam_size, lambda_val=lambda_val,
                    precomputed_dicts=precomputed
                )

                # ── 标准束搜索对比 ──────────────────────────────────────────────
                if run_std:
                    std_paths = mmr_diversity_beam_search(
                        single_outputs, valid_edges_dict, topic_scores,
                        h + 1, K=beam_size, lambda_val=0.0,
                        precomputed_dicts=precomputed
                    )
                    update_std_stats(std_stats, std_paths, gold_ids)

                # 序列化路径（实体名经 resolve 转为可读名）
                mmr_reason_paths = []
                for nodes, rels, score in mmr_paths:
                    mmr_reason_paths.append({
                        "path": [
                            [resolve(data.id2ent[nodes[k]]),
                             data.id2rel[rels[k]],
                             resolve(data.id2ent[nodes[k + 1]])]
                            for k in range(len(rels))
                        ],
                        "log_score": round(float(score), 6),
                    })

                # ── MMR 路径检索指标 ──────────────────────────────────────────
                m, d = update_mmr_stats(mmr_stats, mmr_paths, gold_ids)

                # ── 多阈值对比 ────────────────────────────────────────────────
                update_thresh_stats(thresh_stats, e_score[i], gold_ids, acc_thresholds)

                question_ids = batch[1]['input_ids'][i].tolist()
                question_tokens = data.tokenizer.convert_ids_to_tokens(question_ids)
                question_str = ' '.join(question_tokens).replace(' [PAD]', '').strip()

                gold_ans = [resolve(data.id2ent[x]) for (x, _) in gold_pairs]
                pred_ans = {resolve(data.id2ent[x]): float(f"{y:.3f}")
                            for (x, y) in filter_tensor(e_score[i], 0.9)}

                batch_infos.append({
                    "question": question_str,
                    "topics": topics,
                    "hop": h + 1,
                    "mmr_reason_paths": mmr_reason_paths,
                    "mmr_answer_path_hit": m["answer_hit"],
                    "mmr_top1_hit": m["top1_hit"],
                    "path_diversity": d,
                    "mmr_answer_recall": round(m["recall"], 4),
                    "mmr_precision": round(m["precision"], 4),
                    "mmr_f1": round(m["f1"], 4),
                    "golden": gold_ans,
                    "prediction": pred_ans,
                    "hit": bool(hit_score[i]),
                })

            with open(out_path, 'a', encoding='utf-8') as f:
                for info in batch_infos:
                    f.write(json.dumps(info, ensure_ascii=False) + '\n')
            del batch_infos

            if verbose:
                answers = batch[2]
                for i in range(len(hit_score)):
                    if hit_score[i] == 0:
                        print('================================================================')
                        question_ids = batch[1]['input_ids'][i].tolist()
                        question_tokens = data.tokenizer.convert_ids_to_tokens(question_ids)
                        print(' '.join(question_tokens))
                        topic_id = batch[0][i].argmax(0).item()
                        print('> topic entity: {}'.format(data.id2ent[topic_id]))
                        for t in range(2):
                            print('>>>>>>> step {}'.format(t))
                            tmp = ' '.join(['{}: {:.3f}'.format(x, y) for x, y in
                                zip(question_tokens, outputs['word_attns'][t][i].tolist())])
                            print('> Attention: ' + tmp)
                            print('> Relation:')
                            rel_idx = rel_probs_cpu[t][i].gt(0.9).nonzero().squeeze(1).tolist()
                            for x in rel_idx:
                                print('  {}: {:.3f}'.format(data.id2rel[x], rel_probs_cpu[t][i][x].item()))
                            print('> Entity: {}'.format('; '.join([data.id2ent[_] for _ in
                                ent_probs_cpu[t][i].gt(0.8).nonzero().squeeze(1).tolist()])))
                        print('----')
                        print('> max is {}'.format(data.id2ent[idx[i].item()]))
                        print('> golden: {}'.format('; '.join([data.id2ent[_] for _ in
                            answers[i].gt(0.9).nonzero().squeeze(1).tolist()])))
                        print('> prediction: {}'.format('; '.join([data.id2ent[_] for _ in
                            e_score[i].gt(0.9).nonzero().squeeze(1).tolist()])))
                        print(' '.join(question_tokens))
                        print(outputs['hop_attn'][i].tolist())
                        embed()

            del outputs, hop_attn_cpu, e_score, scores, idx, rel_probs_cpu, ent_probs_cpu

    acc = correct / count
    print_validate_results(acc, hop_count, mmr_stats, thresh_stats, std_stats,
                           run_std, beam_size, lambda_val, acc_thresholds)
    return acc




def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', default='./input')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--mode', default='val', choices=['val', 'vis', 'test', 'train'])
    parser.add_argument('--bert_name', default='bert-base-uncased', choices=['roberta-base', 'bert-base-uncased'])
    parser.add_argument('--beam_size', default=3, type=int,
                        help='MMR beam size')
    parser.add_argument('--lambda_val', default=0.5, type=float,
                        help='MMR lambda (diversity penalty)')
    parser.add_argument('--output_path', default=None,
                        help='Output jsonl path (default: WebQSP/predict_result_webqsp_path_info.jsonl)')
    parser.add_argument('--entity_label', default=None,
                        help='MID→可读名映射文件，JSON 格式 {"m.xxx": "name"}，'
                             '或 TSV 格式 "MID\\tname"。不提供则保留原始 MID。')
    parser.add_argument('--acc_thresholds', default='0.7,0.8,0.9',
                        help='逗号分隔的 e_score 阈值列表，用于多阈值对比')
    parser.add_argument('--no_compare_standard', action='store_true',
                        help='不与标准束搜索（λ=0）做对比')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ent2id, rel2id, triples, train_loader, val_loader = load_data(args.input_dir, args.bert_name, 16)

    model = TransferNet(args, ent2id, rel2id, triples)
    missing, unexpected = model.load_state_dict(torch.load(args.ckpt), strict=False)
    if missing:
        print("Missing keys: {}".format("; ".join(missing)))
    if unexpected:
        print("Unexpected keys: {}".format("; ".join(unexpected)))
    model = model.to(device)
    model.Msubj = model.Msubj.to(device)
    model.Mobj  = model.Mobj.to(device)
    model.Mrel  = model.Mrel.to(device)

    print("[INFO] 预构建边查找字典 (valid_edges_dict) ...", flush=True)
    triples_list = [[int(s), int(r), int(o)] for s, r, o in triples.tolist()]
    valid_edges_dict = build_valid_edges_dict(triples_list)
    print(f"[INFO] 完成，共载入 {len(valid_edges_dict)} 个实体节点的出边。", flush=True)

    # 加载 MID→可读名映射（可选）
    mid2label = None
    if args.entity_label and os.path.exists(args.entity_label):
        print(f"[INFO] 加载实体名称映射: {args.entity_label}", flush=True)
        if args.entity_label.endswith(".json"):
            with open(args.entity_label, encoding="utf-8") as f:
                mid2label = json.load(f)
        else:  # TSV: MID\tname
            mid2label = {}
            with open(args.entity_label, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t", 1)
                    if len(parts) == 2:
                        mid2label[parts[0].strip()] = parts[1].strip()
        print(f"[INFO] 共加载 {len(mid2label)} 条实体名称映射", flush=True)

    loader = val_loader if args.mode in ('val', 'vis', 'test') else train_loader
    acc_thresholds = [float(t) for t in args.acc_thresholds.split(',')]

    verbose = args.mode == 'vis'
    validate(args, model, loader, triples_list, valid_edges_dict, device,
             verbose=verbose, beam_size=args.beam_size, lambda_val=args.lambda_val,
             output_path=args.output_path, mid2label=mid2label,
             acc_thresholds=acc_thresholds,
             compare_standard=not args.no_compare_standard)

if __name__ == '__main__':
    main()
