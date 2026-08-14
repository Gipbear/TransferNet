import torch
import os
import json
import pickle
import numpy as np
from functools import partial
from collections import defaultdict
from transformers import AutoTokenizer
from utils.huggingface import from_pretrained_local_first
from utils.misc import invert_dict

class SparseOneHot:
    """以行内列下标表示的 one-hot 矩阵，展开成稠密张量的动作推迟到目标设备上做。

    CWQ 实体表有 240 万项，稠密 [bsz, num_ents] one-hot 每批约 622 MB；在 collate
    里造三份再逐份拷进显存会让 GPU 长时间空等，这里只搬索引，由使用方在 GPU 上展开。
    """

    __slots__ = ('indices', 'batch_idx', 'batch_size', 'num_ents')

    def __init__(self, indices, batch_idx, batch_size, num_ents):
        self.indices = indices
        self.batch_idx = batch_idx
        self.batch_size = batch_size
        self.num_ents = num_ents

    @classmethod
    def from_rows(cls, rows, num_ents):
        sizes = torch.tensor([row.shape[0] for row in rows])
        batch_idx = torch.repeat_interleave(torch.arange(len(rows)), sizes)
        return cls(torch.cat(rows), batch_idx, len(rows), num_ents)

    def to(self, device, non_blocking=False):
        return SparseOneHot(
            self.indices.to(device, non_blocking=non_blocking),
            self.batch_idx.to(device, non_blocking=non_blocking),
            self.batch_size, self.num_ents,
        )

    def dense(self):
        one_hot = torch.zeros(self.batch_size, self.num_ents, device=self.indices.device)
        one_hot[self.batch_idx, self.indices] = 1
        return one_hot

    def gather_rows(self, cols):
        """取每行在 cols[row] 这一列上的取值，等价于 dense().gather(1, cols[:, None])。"""
        cols = cols.to(self.indices.device)
        hit = torch.zeros(self.batch_size, device=self.indices.device)
        matched = self.indices == cols[self.batch_idx]
        hit[self.batch_idx[matched]] = 1
        return hit

    def __getitem__(self, row):
        """第 row 个样本的列下标，升序返回以对齐稠密 one-hot 上 torch.where 的顺序。"""
        return torch.sort(self.indices[self.batch_idx == row]).values

    def __len__(self):
        return self.batch_size


def collate(batch, num_ents):
    batch = list(zip(*batch))
    topic_entity, question, answer, triples, entity_range = batch
    topic_entity = SparseOneHot.from_rows(topic_entity, num_ents)
    question = {k:torch.cat([q[k] for q in question], dim=0) for k in question[0]}
    answer = SparseOneHot.from_rows(answer, num_ents)
    entity_range = SparseOneHot.from_rows(entity_range, num_ents)
    triple_sizes = torch.tensor([triple.shape[0] for triple in triples])
    triple_batch = torch.repeat_interleave(torch.arange(len(triples)), triple_sizes)
    return topic_entity, question, answer, triples, entity_range, triple_batch


def as_int32(values):
    """把一串实体/关系 id 收成 int32 数组，取代 Python 的 list[int]。

    CWQ 全量 1.48 亿条三元组按 list[list[int]] 存要 187 B/条：每条是一个 list 对象
    (56 B) + 3 个指针 (24 B) + 3 个 int 对象 (各 28 B，实体 id 到 226 万，全都超出
    CPython 的小整数缓存)。换成 int32 后是 14 B/条，25.8 GB 降到 1.9 GB。
    id 上界 226 万，离 int32 的 21 亿很远，不会溢出。
    """
    return np.asarray(values, dtype=np.int32)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, questions, ent2id):
        self.questions = questions
        self.ent2id = ent2id

    def __getitem__(self, index):
        topic_entity, question, answer, triples, entity_range = self.questions[index]
        # as_tensor 对 int32 数组和旧缓存里的 list[int] 都成立，取出来一律是 long，
        # 下游拿到的东西跟以前逐位相同。
        topic_entity = torch.as_tensor(topic_entity, dtype=torch.long)
        answer = torch.as_tensor(answer, dtype=torch.long)
        triples = torch.as_tensor(triples, dtype=torch.long)
        if triples.dim() == 1:
            triples = triples.unsqueeze(0)
        entity_range = torch.as_tensor(entity_range, dtype=torch.long)
        return topic_entity, question, answer, triples, entity_range

    def __len__(self):
        return len(self.questions)

def merge_dev_into_train(train_dataset, dev_dataset, holdout):
    """把 dev 的尾部并入 train，头部 holdout 条留作验证集。

    在已加载的缓存上做纯内存拼接（questions 是普通 list），不重建 3.4 GB 的
    tokenization 缓存。holdout 必须为正且小于 dev 规模：验证集为空时只能按 test
    选 epoch，等于把测试集用作模型选择。
    """
    if holdout <= 0:
        raise ValueError('holdout 必须为正数，否则没有验证集可用于选 epoch')
    if holdout >= len(dev_dataset):
        raise ValueError('holdout {} 不能小于 dev 规模 {}'.format(holdout, len(dev_dataset)))
    val_questions = list(dev_dataset.questions[:holdout])
    merged_questions = list(train_dataset.questions) + list(dev_dataset.questions[holdout:])
    return (Dataset(merged_questions, train_dataset.ent2id),
            Dataset(val_questions, train_dataset.ent2id))


def make_data_loader(dataset, batch_size, training=False, num_workers=0,
                     pin_memory=False, persistent_workers=False, ent2id=None,
                     rel2id=None, tokenizer=None):
    if num_workers < 0:
        raise ValueError('num_workers must be non-negative')
    if persistent_workers and num_workers == 0:
        raise ValueError('persistent_workers requires num_workers > 0')
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        collate_fn=partial(collate, num_ents=len(dataset.ent2id)),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    if ent2id is not None:
        loader.ent2id = ent2id
        loader.id2ent = invert_dict(ent2id)
    if rel2id is not None:
        loader.rel2id = rel2id
        loader.id2rel = invert_dict(rel2id)
    if tokenizer is not None:
        loader.tokenizer = tokenizer
    return loader


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, fn, bert_name, ent2id, rel2id, batch_size, add_rev=False,
                 training=False, num_workers=0, pin_memory=False,
                 persistent_workers=False):
        print('Reading questions from {} {}'.format(fn, '(add reverse)' if add_rev else ''))
        self.tokenizer = from_pretrained_local_first(AutoTokenizer, bert_name)
        self.ent2id = ent2id
        self.rel2id = rel2id
        self.id2ent = invert_dict(ent2id)
        self.id2rel = invert_dict(rel2id)

        data = []
        cnt_bad = 0
        for line in open(fn):
            instance = json.loads(line.strip())

            question = self.tokenizer(instance['question'].strip(), max_length=64, padding='max_length', return_tensors="pt")
            head = instance['entities']
            ans = [ent2id[a['kb_id']] for a in instance['answers']]
            triples = instance['subgraph']['tuples']

            if len(triples) == 0:
                continue

            sub_ents = set(t[0] for t in triples)
            obj_ents = set(t[2] for t in triples)
            entity_range = sub_ents | obj_ents

            is_bad = False
            if all(e not in entity_range for e in head):
                is_bad = True
            if all(e not in entity_range for e in ans):
                is_bad = True

            if is_bad:
                cnt_bad += 1

            if training and is_bad: # skip bad examples during training
                continue

            entity_range = list(entity_range)

            if add_rev:
                supply_triples = []
                # add self relation
                # for e in entity_range:
                #     supply_triples.append([e, self.rel2id['<self>'], e])
                # add reverse relation
                for s, r, o in triples:
                    rev_r = self.rel2id[self.id2rel[r]+'_rev']
                    supply_triples.append([o, rev_r, s])
                triples += supply_triples

            # 就地收成 int32，好让本轮的 Python list 立刻可回收；留到最后统一转换
            # 等于要同时扛住两份，峰值反而更高。
            data.append([as_int32(head), question, as_int32(ans),
                         as_int32(triples), as_int32(entity_range)])

        print('data number: {}, bad number: {} (exluded in training)'.format(len(data), cnt_bad))
        
        dataset = Dataset(data, ent2id)

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=training,
            collate_fn=partial(collate, num_ents=len(dataset.ent2id)),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

# need to download the data from https://github.com/RichardHGL/WSDM2021_NSM
def _cached_datasets(payload):
    ent2id, rel2id, train_data, dev_data, test_data = payload
    cached = (train_data, dev_data, test_data)
    if all(isinstance(data, Dataset) for data in cached):
        return ent2id, rel2id, cached, False
    if all(isinstance(data, torch.utils.data.DataLoader) for data in cached):
        return ent2id, rel2id, tuple(data.dataset for data in cached), True
    raise ValueError('Unsupported CompWebQ cache format; delete the cache and rebuild it')


# 样本字段的存储布局版本。v2 起三元组等 id 序列按 int32 数组存（见 as_int32）。
# 布局进文件名而不是文件内容：v1 缓存里躺着的就是那份 25.8 GB 的 list[list[int]]，
# 想「读进来再判断版本」的话，判断之前内存就已经炸了。
CACHE_LAYOUT = 'v2'


def cache_path(input_dir, bert_name, add_rev=False):
    """缓存文件名带上 bert_name 和存储布局版本。

    缓存里存的是 tokenization 结果，换 encoder 就必须重建。此前所有 encoder 共用
    一个 'cache.pt'，换 encoder 时会静默读到上一个 tokenizer 的结果——不报错，
    只让指标莫名变差，很难联想到是缓存的问题。
    """
    slug = bert_name.replace('/', '_')
    return os.path.join(input_dir, 'cache_{}{}_{}.pt'.format(
        slug, '_rev' if add_rev else '', CACHE_LAYOUT))


def load_data(input_dir, bert_name, batch_size, add_rev=False, num_workers=0,
              pin_memory=False, persistent_workers=False):
    cache_fn = cache_path(input_dir, bert_name, add_rev)
    rev_tag = '_rev' if add_rev else ''
    slug = bert_name.replace('/', '_')
    legacy_fns = [
        # 没记 tokenizer 的远古缓存
        os.path.join(input_dir, 'cache{}.pt'.format(rev_tag)),
        # 记了 tokenizer 但还是 v1 布局（list[int]）的缓存
        os.path.join(input_dir, 'cache_{}{}.pt'.format(slug, rev_tag)),
    ]
    if not os.path.exists(cache_fn):
        for fn in legacy_fns:
            if os.path.exists(fn):
                print('Found stale cache {} (old storage layout); rebuilding as {}. '
                      'Delete it once you no longer need it.'.format(fn, cache_fn))
    if os.path.exists(cache_fn):
        print('Read from cache file: {}'.format(cache_fn))
        with open(cache_fn, 'rb') as fp:
            ent2id, rel2id, datasets, legacy_cache = _cached_datasets(pickle.load(fp))
        if legacy_cache:
            print('Legacy cache contains DataLoader objects; rebuilding loaders with current runtime options')
        train_dataset, dev_dataset, test_dataset = datasets
        tokenizer = from_pretrained_local_first(AutoTokenizer, bert_name)
        print('Train number: {}, dev number: {}, test number: {}'.format(
            len(train_dataset), len(dev_dataset), len(test_dataset)))
        train_data = make_data_loader(
            train_dataset, batch_size, training=True, num_workers=num_workers,
            pin_memory=pin_memory, persistent_workers=persistent_workers,
            ent2id=ent2id, rel2id=rel2id, tokenizer=tokenizer)
        dev_data = make_data_loader(
            dev_dataset, batch_size, num_workers=num_workers,
            pin_memory=pin_memory, ent2id=ent2id, rel2id=rel2id,
            tokenizer=tokenizer)
        test_data = make_data_loader(
            test_dataset, batch_size, num_workers=num_workers,
            pin_memory=pin_memory, ent2id=ent2id, rel2id=rel2id,
            tokenizer=tokenizer)
    else:
        print('Read data...')
        ent2id = {}
        for line in open(os.path.join(input_dir, 'entities.txt')):
            ent2id[line.strip()] = len(ent2id)
        print(len(ent2id))
        rel2id = {}
        for line in open(os.path.join(input_dir, 'relations.txt')):
            rel2id[line.strip()] = len(rel2id)
        # add self relation and reverse relation
        # rel2id['<self>'] = len(rel2id)
        if add_rev:
            for line in open(os.path.join(input_dir, 'relations.txt')):
                rel2id[line.strip()+'_rev'] = len(rel2id)
        print(len(rel2id))

        train_data = DataLoader(
            os.path.join(input_dir, 'train_simple.json'), bert_name, ent2id, rel2id,
            batch_size, add_rev=add_rev, training=True, num_workers=num_workers,
            pin_memory=pin_memory, persistent_workers=persistent_workers)
        dev_data = DataLoader(
            os.path.join(input_dir, 'dev_simple.json'), bert_name, ent2id, rel2id,
            batch_size, add_rev=add_rev, num_workers=num_workers,
            pin_memory=pin_memory)
        test_data = DataLoader(
            os.path.join(input_dir, 'test_simple.json'), bert_name, ent2id, rel2id,
            batch_size, add_rev=add_rev, num_workers=num_workers,
            pin_memory=pin_memory)

        with open(cache_fn, 'wb') as fp:
            pickle.dump((
                ent2id, rel2id, train_data.dataset, dev_data.dataset,
                test_data.dataset,
            ), fp)

    return ent2id, rel2id, train_data, dev_data, test_data



def cnt_hops(input_dir):
    def bfs(triples, start, end):
        if len(start)==0 or len(end)==0:
            return 1000, 1000

        hops = {i:0 for i in start}
        cur_set = set(start)
        next_set = set()
        for h in range(5):
            for s,r,o in triples:
                if s in cur_set and o not in hops:
                    hops[o] = h
                    next_set.add(o)
            cur_set = next_set
            next_set = set()
        only_forwad_res = min(hops.get(i,1000) for i in end)

        hops = {i:0 for i in start}
        cur_set = set(start)
        next_set = set()
        for h in range(5):
            for s,r,o in triples:
                if s in cur_set and o not in hops:
                    hops[o] = h
                    next_set.add(o)
                if o in cur_set and s not in hops:
                    hops[s] = h
                    next_set.add(s)
            cur_set = next_set
            next_set = set()
        add_reverse_res = min(hops.get(i,1000) for i in end)

        return only_forwad_res, add_reverse_res


    ent2id = {}
    for line in open(os.path.join(input_dir, 'entities.txt')):
        ent2id[line.strip()] = len(ent2id)


    for fn in ['train_simple.json', 'test_simple.json']:
        only_forward_res = defaultdict(int)
        add_reverse_res = defaultdict(int)
        for line in open(os.path.join(input_dir, fn)):
            instance = json.loads(line.strip())
            triples = instance['subgraph']['tuples']
            head = instance['entities']
            ans = [ent2id[a['kb_id']] for a in instance['answers']]
            i, j = bfs(triples, head, ans)
            only_forward_res[i] += 1
            add_reverse_res[j] += 1

        print(fn)
        print(only_forward_res)
        print(add_reverse_res) # increase the ratio of 1-hop

if __name__ == '__main__':
    cnt_hops('/data/sjx/dataset/WSDM_processed/CWQ')
