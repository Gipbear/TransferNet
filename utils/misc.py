from collections import defaultdict, Counter, deque
import torch
import json
import pickle
import numpy as np
import torch.nn as nn
import math
from torch.optim.optimizer import Optimizer
import transformers

DUMMY_RELATION = 'DUMMY_RELATION'
DUMMY_ENTITY = 'DUMMY_ENTITY'

DUMMY_RELATION_ID = 0
DUMMY_ENTITY_ID = 0

EPSILON = float(np.finfo(float).eps)
HUGE_INT = 1e31

def batch_device(batch, device):
    res = []
    for x in batch:
        if isinstance(x, torch.Tensor):
            x = x.to(device)
        elif isinstance(x, (dict, transformers.tokenization_utils_base.BatchEncoding)):
            for k in x:
                if isinstance(x[k], torch.Tensor):
                    x[k] = x[k].to(device)
        res.append(x)
    return res

def idx_to_one_hot(idx, size):
    """
    Args:
        idx [bsz, 1]
    Return:
        one_hot [bsz, size]
    """
    one_hot = torch.FloatTensor(len(idx), size)
    one_hot.zero_()
    one_hot.scatter_(1, idx, 1)
    return one_hot

def format_path(path_trace, id2entity, id2relation):
    def get_most_recent_relation(j):
        relation_id = int(path_trace[j][0])
        if relation_id == NO_OP_RELATION_ID:
            return '<null>'
        else:
            return id2relation[relation_id]

    def get_most_recent_entity(j):
        return id2entity[int(path_trace[j][1])]

    path_str = get_most_recent_entity(0)
    for j in range(1, len(path_trace)):
        rel = get_most_recent_relation(j)
        if not rel.endswith('_inv'):
            path_str += ' '+ UseStyle('-{}->'.format(rel), back = 'blue') + ' '
        else:
            path_str += ' ' + UseStyle('<-{}-'.format(rel[:-4]), back = 'green') + ' '
        path_str += get_most_recent_entity(j)
    return path_str

def pad_and_cat(a, padding_value, padding_dim=1):
    max_dim_size = max([x.size()[padding_dim] for x in a])
    padded_a = []
    for x in a:
        if x.size()[padding_dim] < max_dim_size:
            res_len = max_dim_size - x.size()[1]
            pad = nn.ConstantPad1d((0, res_len), padding_value)
            padded_a.append(pad(x))
        else:
            padded_a.append(x)
    return torch.cat(padded_a, dim=0)

def safe_log(x):
    return torch.log(x + EPSILON)

def entropy(p):
    return torch.sum(- p * safe_log(p), 1)

def init_word2id():
    return {
        '<PAD>': 0,
        '<UNK>': 1,
        'E_S': 2,
    }
def init_entity2id():
    return {
        DUMMY_ENTITY: DUMMY_ENTITY_ID
    }
def init_relation2id():
    return {
        DUMMY_RELATION: DUMMY_RELATION_ID
    }

def add_item_to_x2id(item, x2id):
    if not item in x2id:
        x2id[item] = len(x2id)
        
def invert_dict(d):
    return {v: k for k, v in d.items()}

def load_glove(glove_pt, idx_to_token):
    glove = pickle.load(open(glove_pt, 'rb'))
    dim = len(glove['the'])
    matrix = []
    for i in range(len(idx_to_token)):
        token = idx_to_token[i]
        tokens = token.split()
        if len(tokens) > 1:
            v = np.zeros((dim,))
            for token in tokens:
                v = v + glove.get(token, glove['the'])
            v = v / len(tokens)
        else:
            v = glove.get(token, glove['the'])
        matrix.append(v)
    matrix = np.asarray(matrix)
    return matrix

def tile_along_beam(v, beam_size, dim=0):
    """
    Tile a tensor along a specified dimension for the specified beam size.
    :param v: Input tensor.
    :param beam_size: Beam size.
    """
    if dim == -1:
        dim = len(v.size()) - 1
    v = v.unsqueeze(dim + 1)
    v = torch.cat([v] * beam_size, dim=dim+1)
    new_size = []
    for i, d in enumerate(v.size()):
        if i == dim + 1:
            new_size[-1] *= d
        else:
            new_size.append(d)
    return v.view(new_size)

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
                    type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f} ({:.4f})".format(name, meter.median, meter.global_avg)
            )
        return self.delimiter.join(loss_str)


class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        
        self.degenerated_to_sgd = degenerated_to_sgd
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss



STYLE = {
        'fore':
        {   # 前景色
            'black'    : 30,   #  黑色
            'red'      : 31,   #  红色
            'green'    : 32,   #  绿色
            'yellow'   : 33,   #  黄色
            'blue'     : 34,   #  蓝色
            'purple'   : 35,   #  紫红色
            'cyan'     : 36,   #  青蓝色
            'white'    : 37,   #  白色
        },

        'back' :
        {   # 背景
            'black'     : 40,  #  黑色
            'red'       : 41,  #  红色
            'green'     : 42,  #  绿色
            'yellow'    : 43,  #  黄色
            'blue'      : 44,  #  蓝色
            'purple'    : 45,  #  紫红色
            'cyan'      : 46,  #  青蓝色
            'white'     : 47,  #  白色
        },

        'mode' :
        {   # 显示模式
            'mormal'    : 0,   #  终端默认设置
            'bold'      : 1,   #  高亮显示
            'underline' : 4,   #  使用下划线
            'blink'     : 5,   #  闪烁
            'invert'    : 7,   #  反白显示
            'hide'      : 8,   #  不可见
        },

        'default' :
        {
            'end' : 0,
        },
}

def UseStyle(string, mode = None, fore = None, back = None):

    mode  = '%s' % STYLE['mode'][mode]  if mode else ''

    fore  = '%s' % STYLE['fore'][fore] if fore else ''

    back  = '%s' % STYLE['back'][back] if back else ''

    style = ';'.join([s for s in [mode, fore, back] if s])

    style = '\033[%sm' % style if style else ''

    end   = '\033[%sm' % STYLE['default']['end'] if style else ''

    return '%s%s%s' % (style, string, end)
