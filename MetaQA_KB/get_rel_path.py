import os
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from tqdm import tqdm
from utils.misc import invert_dict

input_dir = "data/input/MetaQA_KB"
Msubj = np.load(os.path.join(input_dir, 'Msubj.npy'))
Mobj = np.load(os.path.join(input_dir, 'Mobj.npy'))
Mrel = np.load(os.path.join(input_dir, 'Mrel.npy'))
triples = np.stack([Msubj[:, 1], Mrel[:, 1], Mobj[:, 1]], axis=1).tolist()
triples = [tuple(x) for x in triples]


def load_vocab(path):
    vocab = json.load(open(path))
    vocab['id2word'] = invert_dict(vocab['word2id'])
    vocab['id2entity'] = invert_dict(vocab['entity2id'])
    vocab['id2relation'] = invert_dict(vocab['relation2id'])
    return vocab


vocab = load_vocab(os.path.join(input_dir, 'vocab.json'))


def get_valid_path(paths: list[list[int, int, int]], hop: int):
    truth_paths = []
    a, b = 0, 0
    for i in range(hop):
        # valid_paths = [row for row in paths[i] if tuple(row) in triples]
        valid_paths = []
        for row in paths[i]:
            if tuple(row) in triples:
                valid_paths.append([vocab['id2entity'][row[0]], vocab['id2relation']
                                   [row[1]], vocab['id2entity'][row[2]]])
        truth_paths.append(valid_paths)
        a += len(paths[i])
        b += len(valid_paths)
    if a != b:
        print(a, b)
    else:
        print("ok")
    return truth_paths


args = []
with open("MetaQA_KB/predict_result_with_path_info.jsonl") as f:
    for line in tqdm(f.readlines(), desc="Loading data"):
        data = json.loads(line.strip())
        reason_paths = data['reason_paths']
        hop = data['hop']
        args.append((reason_paths, hop))
        # truth_paths = get_valid_path(reason_paths, hop)
print('start processing...')
with ProcessPoolExecutor(max_workers=15) as executor:
    results = list(tqdm(executor.map(get_valid_path, *zip(*args)), total=len(args), desc="Processing paths"))
    # results = list(tqdm(executor.map(get_valid_path, *zip(*args[:100])), total=len(args), desc="Processing paths"))

truth_paths_data = []
with open("MetaQA_KB/predict_result_with_path_info.jsonl") as f:
    idx = 0
    for line in tqdm(f.readlines(), desc="Loading data"):
        data = json.loads(line.strip())
        data['truth_paths'] = results[idx]
        data.pop('reason_paths')
        truth_paths_data.append(data)
        idx += 1
        # if idx == 100:
        #     break
print('finish processing...')
with open("MetaQA_KB/predict_result_with_truth_path.jsonl", 'w') as f:
    for info in truth_paths_data:
        f.write(json.dumps(info) + '\n')
print("Saved to MetaQA_KB/predict_result_with_truth_path.jsonl")
