import os
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from collections import defaultdict
from utils.misc import batch_device
from .data import load_data
from .model import TransferNet

from IPython import embed

def filter_tensor(tensor, threshold=0.9):
    indices = torch.where(tensor >= threshold)[0]
    scores = tensor[indices]
    return list(zip(indices.tolist(), scores.tolist()))


def validate(args, model, data, device, verbose = False, get_path=True):
    model.eval()
    count = 0
    correct = 0
    hop_count = defaultdict(list)
    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            outputs = model(*batch_device(batch, device)) # [bsz, Esize]
            e_score = outputs['e_score'].cpu()
            scores, idx = torch.max(e_score, dim = 1) # [bsz], [bsz]
            match_score = torch.gather(batch[2], 1, idx.unsqueeze(-1)).squeeze().tolist()
            count += len(match_score)
            correct += sum(match_score)
            for i in range(len(match_score)):
                h = outputs['hop_attn'][i].argmax().item()
                hop_count[h].append(match_score[i])
            if get_path:
                for i in range(len(match_score)):
                    question_ids = batch[1]['input_ids'][i].tolist()
                    question_tokens = data.tokenizer.convert_ids_to_tokens(question_ids)
                    question = ' '.join([token for token in question_tokens if token not in ['[CLS]', '[SEP]', '[PAD]']])
                    topic_scores = filter_tensor(batch[0][i], 1)
                    topics = [data.id2ent[x] for (x, _) in topic_scores]
                    hop = outputs['hop_attn'][i].argmax().item()
                    relation_list, entity_list = [], []
                    subj_ids = [x for (x, _) in topic_scores]
                    reason_paths = []
                    for t in range(hop+1):
                        relation_scores = filter_tensor(outputs['rel_probs'][t][i], 0.9)
                        relation_dict = {data.id2rel[x]: float(f"{y:.3f}") for (x, y) in relation_scores}
                        entity_scores = filter_tensor(outputs['ent_probs'][t][i], 0.9)
                        entity_dict = {data.id2ent[x]: float(f"{y:.3f}") for( x, y) in entity_scores}
                        relation_list.append(relation_dict)
                        entity_list.append(entity_dict)
                        # 判断 subj_id, relation_id, obj_id 是否在知识图谱中
                        paths = []
                        for subj_id in subj_ids:
                            for (rel_id, _) in relation_scores:
                                for (obj_id, _) in entity_scores:
                                    paths.append((subj_id, rel_id, obj_id))
                        subj_ids = [obj_id for (obj_id, _) in entity_scores]
                        # valid_paths = model.kg.get_valid_path(paths)
                        # print(valid_paths)
                        reason_paths.append(paths)

                    gold_ans = [data.id2ent[x] for (x, _) in filter_tensor(batch[2][i], 1)]
                    pred_ans = {data.id2ent[x]: float(f"{y:.3f}") for (x, y) in filter_tensor(e_score[i], 0.9)}
                    info = {
                        "question": question,
                        "topics": topics,
                        "hop": hop,
                        "relation": relation_list,
                        "entity": entity_list,
                        "reason_paths": reason_paths,
                        "golden": gold_ans,
                        "prediction": pred_ans,
                    }
                    # all_path_info.append(info)
                    # print(all_path_info)
                    # print('> golden: {}'.format('; '.join([data.id2ent[_]
                    #       for _ in range(len(answers[i])) if answers[i][_].item() == 1])))
                    # print('> prediction: {}'.format('; '.join([data.id2ent[_]
                    #       for _ in range(len(answers[i])) if e_score[i][_].item() > 0.9])))
                    import json
                    with open("../autodl-tmp/CWQ/predict_result_with_path_info_val.jsonl", 'a') as f:
                        f.write(json.dumps(info) + '\n')
            if verbose:
                answers = batch[2]
                for i in range(len(match_score)):
                    if match_score[i] == 0:
                        print('================================================================')
                        question_ids = batch[1]['input_ids'][i].tolist()
                        question_tokens = data.tokenizer.convert_ids_to_tokens(question_ids)
                        print(' '.join(question_tokens))
                        topic_id = batch[0][i].argmax(0).item()
                        print('> topic entity: {}'.format(data.id2ent[topic_id]))
                        for t in range(2):
                            print('>>>>>>> step {}'.format(t))
                            tmp = ' '.join(['{}: {:.3f}'.format(x, y) for x,y in 
                                zip(question_tokens, outputs['word_attns'][t][i].tolist())])
                            print('> Attention: ' + tmp)
                            print('> Relation:')
                            rel_idx = outputs['rel_probs'][t][i].gt(0.9).nonzero().squeeze(1).tolist()
                            for x in rel_idx:
                                print('  {}: {:.3f}'.format(data.id2rel[x], outputs['rel_probs'][t][i][x].item()))

                            print('> Entity: {}'.format('; '.join([data.id2ent[_] for _ in outputs['ent_probs'][t][i].gt(0.8).nonzero().squeeze(1).tolist()])))
                        print('----')
                        print('> max is {}'.format(data.id2ent[idx[i].item()]))
                        print('> golden: {}'.format('; '.join([data.id2ent[_] for _ in answers[i].gt(0.9).nonzero().squeeze(1).tolist()])))
                        print('> prediction: {}'.format('; '.join([data.id2ent[_] for _ in e_score[i].gt(0.9).nonzero().squeeze(1).tolist()])))
                        print(' '.join(question_tokens))
                        print(outputs['hop_attn'][i].tolist())
                        embed()
    acc = correct / count
    print(acc)
    print('pred hop accuracy: 1-hop {} (total {}), 2-hop {} (total {})'.format(
        sum(hop_count[0])/(len(hop_count[0])+0.1),
        len(hop_count[0]),
        sum(hop_count[1])/(len(hop_count[1])+0.1),
        len(hop_count[1]),
        ))
    return acc


def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', default = './input')
    parser.add_argument('--ckpt', required = True)
    parser.add_argument('--mode', default='val', choices=['val', 'vis', 'test', 'train'])
    parser.add_argument('--num_ways', default=1, type=int)
    parser.add_argument('--num_steps', default=2, type=int)
    parser.add_argument('--bert_name', default='bert-base-cased', choices=['roberta-base', 'bert-base-cased', 'bert-base-uncased'])
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ent2id, rel2id, train_loader, val_loader, test_loader = load_data(args.input_dir, args.bert_name, 16)

    model = TransferNet(args, ent2id, rel2id)
    missing, unexpected = model.load_state_dict(torch.load(args.ckpt), strict=False)
    if missing:
        print("Missing keys: {}".format("; ".join(missing)))
    if unexpected:
        print("Unexpected keys: {}".format("; ".join(unexpected)))
    model = model.to(device)
    # model.triples = model.triples.to(device)
    # model.Msubj = model.Msubj.to(device)
    # model.Mobj = model.Mobj.to(device)
    # model.Mrel = model.Mrel.to(device)

    if args.mode == 'vis':
        validate(args, model, test_loader, device, True)
    elif args.mode == 'val':
        validate(args, model, val_loader, device, False)
    elif args.mode == 'test':
        validate(args, model, test_loader, device, False)
    elif args.mode == 'train':
        validate(args, model, train_loader, device, False)

if __name__ == '__main__':
    main()
