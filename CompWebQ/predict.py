import json
import os
import torch
import argparse
from tqdm import tqdm
from collections import defaultdict
from utils.misc import batch_device
from utils.path_utils import filter_tensor, mmr_diversity_beam_search, build_valid_edges_dict
from utils.eval_utils import (
    create_mmr_stats, create_thresh_stats, create_std_stats,
    update_mmr_stats, update_thresh_stats, update_std_stats,
    print_validate_results,
)
from .data import load_data
from .model import TransferNet

from IPython import embed


def validate(args, model, data, device, verbose=False,
             beam_size=3, lambda_val=0.5, output_path=None,
             acc_thresholds=None, compare_standard=True):
    if acc_thresholds is None:
        acc_thresholds = [0.7, 0.8, 0.9]

    thresh_stats = create_thresh_stats(acc_thresholds)
    run_std = compare_standard and (lambda_val != 0.0)
    std_stats = create_std_stats()

    model.eval()
    count = 0
    correct = 0
    hop_count = defaultdict(list)
    mmr_stats = create_mmr_stats()

    out_path = output_path or "CompWebQ/predict_result_cwq_path_info.jsonl"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    open(out_path, 'w').close()

    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            outputs = model(*batch_device(batch, device))  # [bsz, Esize]
            e_score = outputs['e_score'].cpu()
            scores, idx = torch.max(e_score, dim=1)
            match_score = torch.gather(batch[2], 1, idx.unsqueeze(-1)).squeeze().tolist()
            count += len(match_score)
            correct += sum(match_score)

            hop_attn_cpu  = outputs['hop_attn'].cpu()
            rel_probs_cpu = [t.cpu() for t in outputs['rel_probs']]
            ent_probs_cpu = [t.cpu() for t in outputs['ent_probs']]

            batch_infos = []
            for i in range(len(match_score)):
                h = hop_attn_cpu[i].argmax().item()
                hop_count[h].append(match_score[i])

                topic_scores = filter_tensor(batch[0][i], 1)
                gold_pairs   = filter_tensor(batch[2][i], 0.5)
                gold_ids     = {x for (x, _) in gold_pairs}

                # CWQ: triples 是逐样本的，从 batch[3][i] 动态构建
                sample_triples = batch[3][i].tolist()
                valid_edges_dict = build_valid_edges_dict(sample_triples)

                single_outputs = {
                    'rel_probs': [rel_probs_cpu[t][i] for t in range(len(rel_probs_cpu))],
                    'ent_probs': [ent_probs_cpu[t][i] for t in range(len(ent_probs_cpu))],
                }

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

                if run_std:
                    std_paths = mmr_diversity_beam_search(
                        single_outputs, valid_edges_dict, topic_scores,
                        h + 1, K=beam_size, lambda_val=0.0,
                        precomputed_dicts=precomputed
                    )
                    update_std_stats(std_stats, std_paths, gold_ids)

                mmr_reason_paths = []
                for nodes, rels, score in mmr_paths:
                    mmr_reason_paths.append({
                        "path": [
                            [data.id2ent[nodes[k]],
                             data.id2rel[rels[k]],
                             data.id2ent[nodes[k + 1]]]
                            for k in range(len(rels))
                        ],
                        "log_score": round(float(score), 6),
                    })

                path_m, path_d = update_mmr_stats(mmr_stats, mmr_paths, gold_ids)
                update_thresh_stats(thresh_stats, e_score[i], gold_ids, acc_thresholds)

                question_ids = batch[1]['input_ids'][i].tolist()
                question_tokens = data.tokenizer.convert_ids_to_tokens(question_ids)
                question_str = ' '.join(question_tokens).replace(' [PAD]', '').strip()

                gold_ans = [data.id2ent[x] for (x, _) in gold_pairs]
                pred_ans = {data.id2ent[x]: float(f"{y:.3f}")
                            for (x, y) in filter_tensor(e_score[i], 0.9)}

                batch_infos.append({
                    "question": question_str,
                    "hop": h + 1,
                    "mmr_reason_paths": mmr_reason_paths,
                    "mmr_answer_path_hit": path_m["answer_hit"],
                    "mmr_top1_hit": path_m["top1_hit"],
                    "path_diversity": path_d,
                    "mmr_answer_recall": round(path_m["recall"], 4),
                    "mmr_precision": round(path_m["precision"], 4),
                    "mmr_f1": round(path_m["f1"], 4),
                    "golden": gold_ans,
                    "prediction": pred_ans,
                    "hit": bool(match_score[i]),
                })

                if verbose and match_score[i] == 0:
                    print('================================================================')
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
                        batch[2][i].gt(0.9).nonzero().squeeze(1).tolist()])))
                    print('> prediction: {}'.format('; '.join([data.id2ent[_] for _ in
                        e_score[i].gt(0.9).nonzero().squeeze(1).tolist()])))
                    print(' '.join(question_tokens))
                    print(outputs['hop_attn'][i].tolist())
                    embed()

            with open(out_path, 'a', encoding='utf-8') as f:
                for info in batch_infos:
                    f.write(json.dumps(info, ensure_ascii=False) + '\n')
            del batch_infos

            del outputs, hop_attn_cpu, e_score, scores, idx, rel_probs_cpu, ent_probs_cpu

    acc = correct / count
    print_validate_results(acc, hop_count, mmr_stats, thresh_stats, std_stats,
                           run_std, beam_size, lambda_val, acc_thresholds)
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='./input')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--mode', default='val', choices=['val', 'vis', 'test'])
    parser.add_argument('--bert_name', default='bert-base-cased', choices=['roberta-base', 'bert-base-cased', 'bert-base-uncased'])
    parser.add_argument('--num_steps', default=2, type=int)
    parser.add_argument('--num_ways', default=1, type=int)
    parser.add_argument('--beam_size', default=3, type=int)
    parser.add_argument('--lambda_val', default=0.5, type=float)
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--acc_thresholds', default='0.7,0.8,0.9')
    parser.add_argument('--no_compare_standard', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ent2id, rel2id, triples, train_loader, val_loader = load_data(
        args.input_dir, args.bert_name, 16)

    model = TransferNet(args, ent2id, rel2id)
    missing, unexpected = model.load_state_dict(torch.load(args.ckpt), strict=False)
    if missing:
        print("Missing keys: {}".format("; ".join(missing)))
    if unexpected:
        print("Unexpected keys: {}".format("; ".join(unexpected)))
    model = model.to(device)

    loader = val_loader if args.mode in ('val', 'vis') else val_loader
    acc_thresholds = [float(t) for t in args.acc_thresholds.split(',')]
    verbose = args.mode == 'vis'

    validate(args, model, loader, device,
             verbose=verbose, beam_size=args.beam_size, lambda_val=args.lambda_val,
             output_path=args.output_path, acc_thresholds=acc_thresholds,
             compare_standard=not args.no_compare_standard)

if __name__ == '__main__':
    main()
