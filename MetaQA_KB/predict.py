import json
import os
import numpy as np
import torch
import argparse
from tqdm import tqdm
from collections import defaultdict
from utils.misc import idx_to_one_hot
from utils.path_utils import filter_tensor, mmr_diversity_beam_search, build_valid_edges_dict
from utils.eval_utils import (
    create_mmr_stats, create_thresh_stats, create_std_stats,
    update_mmr_stats, update_thresh_stats, update_std_stats,
    print_validate_results,
)
from .data import DataLoader
from .model import TransferNet

from IPython import embed


def validate(args, model, data, valid_edges_dict, device, verbose=False,
             beam_size=3, lambda_val=0.5, output_path=None,
             acc_thresholds=None, compare_standard=True):
    if acc_thresholds is None:
        acc_thresholds = [0.7, 0.8, 0.9]

    vocab = data.vocab
    thresh_stats = create_thresh_stats(acc_thresholds)
    run_std = compare_standard and (lambda_val != 0.0)
    std_stats = create_std_stats()

    model.eval()
    count = defaultdict(int)
    correct = defaultdict(int)
    hop_count = defaultdict(list)
    mmr_stats = create_mmr_stats()

    out_path = output_path or "MetaQA_KB/predict_result_metaqa_path_info.jsonl"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    open(out_path, 'w').close()  # 清空旧内容

    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            questions, topic_entities, answers, hops = batch
            topic_entities_onehot = idx_to_one_hot(topic_entities, len(vocab['entity2id']))
            answers_onehot = idx_to_one_hot(answers, len(vocab['entity2id']))
            answers_onehot[:, 0] = 0  # 排除 DUMMY_ENTITY
            questions = questions.to(device)
            topic_entities_onehot = topic_entities_onehot.to(device)
            hops_list = hops.tolist()

            outputs = model(questions, topic_entities_onehot)  # [bsz, Esize]
            e_score = outputs['e_score'].cpu()
            scores, idx = torch.max(e_score, dim=1)  # [bsz], [bsz]
            match_score = torch.gather(answers_onehot, 1, idx.unsqueeze(-1)).squeeze().tolist()

            rel_probs_cpu = [t.cpu() for t in outputs['rel_probs']]
            ent_probs_cpu = [t.cpu() for t in outputs['ent_probs']]

            batch_infos = []
            for i, (h_val, m) in enumerate(zip(hops_list, match_score)):
                count['all'] += 1
                count['{}-hop'.format(h_val)] += 1
                correct['all'] += m
                correct['{}-hop'.format(h_val)] += m

                # hop 数：MetaQA 直接使用 ground truth hop（h_val），从 1 开始
                h = h_val  # 1-indexed hop count

                # topic_scores: [(entity_id, score)] 格式，one-hot 中 score=1
                topic_onehot_i = topic_entities_onehot[i].cpu()
                topic_scores = filter_tensor(topic_onehot_i, 1)

                # gold_ids：从 one-hot answers 中提取
                gold_ids = set(answers_onehot[i].gt(0.5).nonzero().squeeze(1).tolist())

                hop_count[h - 1].append(m)  # hop_count 按 0-indexed hop 存储

                single_outputs = {
                    'rel_probs': [rel_probs_cpu[t][i] for t in range(len(rel_probs_cpu))],
                    'ent_probs': [ent_probs_cpu[t][i] for t in range(len(ent_probs_cpu))],
                }

                # 预计算 rel_dict / ent_dict，MMR 和标准束搜索共享
                precomputed = [
                    (dict(filter_tensor(single_outputs['rel_probs'][t], 0.01)),
                     dict(filter_tensor(single_outputs['ent_probs'][t], 0.01)))
                    for t in range(h)
                ]

                mmr_paths = mmr_diversity_beam_search(
                    single_outputs, valid_edges_dict, topic_scores,
                    h, K=beam_size, lambda_val=lambda_val,
                    precomputed_dicts=precomputed
                )

                # ── 标准束搜索对比 ──────────────────────────────────────────────
                if run_std:
                    std_paths = mmr_diversity_beam_search(
                        single_outputs, valid_edges_dict, topic_scores,
                        h, K=beam_size, lambda_val=0.0,
                        precomputed_dicts=precomputed
                    )
                    update_std_stats(std_stats, std_paths, gold_ids)

                # 序列化路径
                mmr_reason_paths = []
                for nodes, rels, score in mmr_paths:
                    mmr_reason_paths.append({
                        "path": [
                            [vocab['id2entity'][nodes[k]],
                             vocab['id2relation'][rels[k]],
                             vocab['id2entity'][nodes[k + 1]]]
                            for k in range(len(rels))
                        ],
                        "log_score": round(float(score), 6),
                    })

                # ── MMR 路径检索指标 ──────────────────────────────────────────
                path_m, path_d = update_mmr_stats(mmr_stats, mmr_paths, gold_ids)

                # ── 多阈值对比 ────────────────────────────────────────────────
                update_thresh_stats(thresh_stats, e_score[i], gold_ids, acc_thresholds)

                question_str = ' '.join([vocab['id2word'][w] for w in questions[i].cpu().tolist() if w > 0])
                gold_ans = [vocab['id2entity'][x] for x in gold_ids]
                pred_ans = {vocab['id2entity'][x]: float(f"{y:.3f}")
                            for (x, y) in filter_tensor(e_score[i], 0.9)}

                batch_infos.append({
                    "question": question_str,
                    "hop": h,
                    "mmr_reason_paths": mmr_reason_paths,
                    "mmr_answer_path_hit": path_m["answer_hit"],
                    "mmr_top1_hit": path_m["top1_hit"],
                    "path_diversity": path_d,
                    "mmr_answer_recall": round(path_m["recall"], 4),
                    "mmr_precision": round(path_m["precision"], 4),
                    "mmr_f1": round(path_m["f1"], 4),
                    "golden": gold_ans,
                    "prediction": pred_ans,
                    "hit": bool(m),
                })

                if verbose and h_val == 3:
                    print('================================================================')
                    print(question_str)
                    print('hop: {}'.format(h_val))
                    topic_id = topic_entities_onehot[i].max(0)[1].item()
                    print('> topic entity: {}'.format(vocab['id2entity'][topic_id]))
                    for t in range(args.num_steps):
                        print('> > > step {}'.format(t))
                        tmp = ' '.join(['{}: {:.3f}'.format(vocab['id2word'][x], y) for x, y in
                            zip(questions.tolist()[i], outputs['word_attns'][t].tolist()[i])
                            if x > 0])
                        print('> ' + tmp)
                        tmp = ' '.join(['{}: {:.3f}'.format(vocab['id2relation'][x], y) for x, y in
                            enumerate(outputs['rel_probs'][t].tolist()[i])])
                        print('> ' + tmp)
                        print('> entity: {}'.format('; '.join([vocab['id2entity'][_] for _ in range(len(answers_onehot[i])) if outputs['ent_probs'][t][i][_].item() > 0.9])))
                    print('----')
                    print('> max is {}'.format(vocab['id2entity'][idx[i].item()]))
                    print('> golden: {}'.format('; '.join([vocab['id2entity'][_] for _ in range(len(answers_onehot[i])) if answers_onehot[i][_].item() == 1])))
                    print('> prediction: {}'.format('; '.join([vocab['id2entity'][_] for _ in range(len(answers_onehot[i])) if e_score[i][_].item() > 0.9])))
                    embed()

            with open(out_path, 'a', encoding='utf-8') as f:
                for info in batch_infos:
                    f.write(json.dumps(info, ensure_ascii=False) + '\n')
            del batch_infos

            del outputs, e_score, scores, idx, rel_probs_cpu, ent_probs_cpu

    acc_dict = {k: correct[k] / count[k] for k in count}
    result = ' | '.join(['%s:%.4f' % (key, value) for key, value in acc_dict.items()])
    print(result)

    # 将 acc_dict 转为与 print_validate_results 兼容的 scalar acc（用 all 分区）
    acc = acc_dict.get('all', 0.0)
    print_validate_results(acc, hop_count, mmr_stats, thresh_stats, std_stats,
                           run_std, beam_size, lambda_val, acc_thresholds)
    return acc_dict


def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', default='./input')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--mode', default='val', choices=['val', 'vis', 'test'])
    # model hyperparameters
    parser.add_argument('--num_steps', default=3, type=int)
    parser.add_argument('--dim_word', default=300, type=int)
    parser.add_argument('--dim_hidden', default=1024, type=int)
    parser.add_argument('--aux_hop', type=int, default=1, choices=[0, 1],
                        help='utilize question hop to constrain the probability of self relation')
    # MMR 评测参数
    parser.add_argument('--beam_size', default=3, type=int,
                        help='MMR beam size')
    parser.add_argument('--lambda_val', default=0.5, type=float,
                        help='MMR lambda (diversity penalty)')
    parser.add_argument('--output_path', default=None,
                        help='Output jsonl path (default: MetaQA_KB/predict_result_metaqa_path_info.jsonl)')
    parser.add_argument('--acc_thresholds', default='0.7,0.8,0.9',
                        help='逗号分隔的 e_score 阈值列表，用于多阈值对比')
    parser.add_argument('--no_compare_standard', action='store_true',
                        help='不与标准束搜索（λ=0）做对比')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    val_pt = os.path.join(args.input_dir, 'val.pt')
    test_pt = os.path.join(args.input_dir, 'test.pt')
    val_loader = DataLoader(vocab_json, val_pt, 64, True)
    test_loader = DataLoader(vocab_json, test_pt, 64)
    vocab = val_loader.vocab

    model = TransferNet(args, args.dim_word, args.dim_hidden, vocab)
    missing, unexpected = model.load_state_dict(torch.load(args.ckpt), strict=False)
    if missing:
        print("Missing keys: {}".format("; ".join(missing)))
    if unexpected:
        print("Unexpected keys: {}".format("; ".join(unexpected)))
    model = model.to(device)
    model.kg.Msubj = model.kg.Msubj.to(device)
    model.kg.Mobj = model.kg.Mobj.to(device)
    model.kg.Mrel = model.kg.Mrel.to(device)

    num_params = sum(np.prod(p.size()) for p in model.parameters())
    print('number of parameters: {}'.format(num_params))

    # 从 npy 文件重建 triples_list，构建边查找字典
    print("[INFO] 预构建边查找字典 (valid_edges_dict) ...", flush=True)
    Msubj_arr = np.load(os.path.join(args.input_dir, 'Msubj.npy'))
    Mobj_arr  = np.load(os.path.join(args.input_dir, 'Mobj.npy'))
    Mrel_arr  = np.load(os.path.join(args.input_dir, 'Mrel.npy'))
    # 每行格式：[triple_idx, entity/rel_id]，取第二列重建 (subj, rel, obj) 三元组
    triples_list = np.stack([Msubj_arr[:, 1], Mrel_arr[:, 1], Mobj_arr[:, 1]], axis=1).tolist()
    triples_list = [[int(s), int(r), int(o)] for s, r, o in triples_list]
    valid_edges_dict = build_valid_edges_dict(triples_list)
    print(f"[INFO] 完成，共载入 {len(valid_edges_dict)} 个实体节点的出边。", flush=True)

    acc_thresholds = [float(t) for t in args.acc_thresholds.split(',')]
    loader = val_loader if args.mode in ('val', 'vis') else test_loader
    verbose = args.mode == 'vis'

    validate(args, model, loader, valid_edges_dict, device,
             verbose=verbose, beam_size=args.beam_size, lambda_val=args.lambda_val,
             output_path=args.output_path, acc_thresholds=acc_thresholds,
             compare_standard=not args.no_compare_standard)

if __name__ == '__main__':
    main()
