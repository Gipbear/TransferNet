import collections
import os
import pickle
from collections import defaultdict
import torch
import torch.nn as nn
from utils.misc import *
import numpy as np


class KnowledgeGraph(nn.Module):
    def __init__(self, args, vocab):
        super(KnowledgeGraph, self).__init__()
        self.args = args
        self.entity2id, self.id2entity = vocab['entity2id'], vocab['id2entity']
        self.relation2id, self.id2relation = vocab['relation2id'], vocab['id2relation']
        Msubj = torch.from_numpy(np.load(os.path.join(args.input_dir, 'Msubj.npy'))).long()
        Mobj = torch.from_numpy(np.load(os.path.join(args.input_dir, 'Mobj.npy'))).long()
        Mrel = torch.from_numpy(np.load(os.path.join(args.input_dir, 'Mrel.npy'))).long()
        Tsize = Msubj.size()[0]
        Esize = len(self.entity2id)
        Rsize = len(self.relation2id)
        # 生成混合张量，在对应 t_id, e/r_id 上使用 1 进行标记
        self.Msubj = torch.sparse_coo_tensor(Msubj.t(), torch.FloatTensor([1] * Tsize), torch.Size([Tsize, Esize]))
        self.Mobj = torch.sparse_coo_tensor(Mobj.t(), torch.FloatTensor([1] * Tsize), torch.Size([Tsize, Esize]))
        self.Mrel = torch.sparse_coo_tensor(Mrel.t(), torch.FloatTensor([1] * Tsize), torch.Size([Tsize, Rsize]))
        self.num_entities = len(self.entity2id)
        triples = torch.stack([Msubj[:, 1], Mrel[:, 1], Mobj[:, 1]], axis=1).tolist()
        self.Triples = [tuple(x) for x in triples]

    def get_valid_path(self, paths: list[list[int, int, int]]):
        # todo: 先记录下来
        return paths
        valid_paths = [row for row in paths if row in self.Triples]
        return valid_paths

    #     self.Triples = torch.stack([Msubj[:, 1], Mrel[:, 1], Mobj[:, 1]], axis=1).unsqueeze(0) # [1, Tsize, 3]

    # def get_valid_path(self, paths: list[list[int, int, int]]):
    #     path_tensor = torch.tensor(paths).unsqueeze(1)  # [m, 1, 3]
    #     matches = (self.Triples == path_tensor).all(dim=-1)  # [m, Tsize]
    #     result = matches.any(dim=1)  # [m]
    #     valid_paths = path_tensor[result].tolist()
    #     return valid_paths
