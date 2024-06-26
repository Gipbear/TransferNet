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
 
