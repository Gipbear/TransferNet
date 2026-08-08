import torch
import torch.nn as nn
import math
from transformers import AutoModel
from utils.BiGRU import GRU, BiGRU
from utils.huggingface import from_pretrained_local_first
from .data import SparseOneHot


def propagate_triples(last_e, rel_dist, triples, triple_batch, num_ents):
    sub, rel, obj = triples[:, 0], triples[:, 1], triples[:, 2]
    contributions = last_e[triple_batch, sub] * rel_dist[triple_batch, rel]
    target = triple_batch * num_ents + obj
    return last_e.new_zeros(last_e.shape[0] * num_ents).index_add_(
        0, target, contributions).view(last_e.shape[0], num_ents)


class TransferNet(nn.Module):
    def __init__(self, args, ent2id, rel2id):
        super().__init__()
        num_relations = len(rel2id)
        self.num_ents = len(ent2id)
        self.num_steps = args.num_steps
        self.num_ways = args.num_ways
        # 官方常数 9 是按 WebQSP 定的。CWQ 子图中位 1240 个实体、答案常常只有 1 个，
        # 正样本占比约 0.067%，放大 10 倍后负样本总权重仍是正样本的约 150 倍。
        self.pos_weight = getattr(args, 'pos_weight', 9)
        self.use_stay_gate = getattr(args, 'stay_gate', False)
        self.score_norm = getattr(args, 'score_norm', 'elem')
        dropout_p = getattr(args, 'dropout', 0.0)
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

        self.bert_encoder = from_pretrained_local_first(
            AutoModel, args.bert_name, return_dict=True
        )
        dim_hidden = self.bert_encoder.config.hidden_size

        self.step_encoders = {}
        self.hop_selectors = {}
        self.rel_classifiers = {}
        self.stay_gates = {}
        for i in range(self.num_ways):
            for j in range(self.num_steps):
                m = nn.Sequential(
                    nn.Linear(dim_hidden*2, dim_hidden),
                    nn.Tanh()
                )
                name = 'way_{}_step_{}'.format(i, j)
                self.step_encoders[name] = m
                self.add_module(name, m)

            m = nn.Linear(dim_hidden, self.num_steps)
            self.hop_selectors['way_{}'.format(i)] = m
            self.add_module('hop-way_{}'.format(i), m)

            m = nn.Linear(dim_hidden, num_relations)
            self.rel_classifiers['way_{}'.format(i)] = m
            self.add_module('rel-way_{}'.format(i), m)

            if self.use_stay_gate:
                # 可学习的 self-loop：原实现里 <self> 关系被注释掉了，直接补 self 三元组
                # 每样本要多约 1240 条边，这里改成按步生成的停留比例，显存不变。
                m = nn.Linear(dim_hidden, 1)
                self.stay_gates['way_{}'.format(i)] = m
                self.add_module('stay-way_{}'.format(i), m)
        


    def forward(self, heads, questions, answers=None, triples=None, entity_range=None,
                triple_batch=None):
        # one-hot 在这里才展开，稠密矩阵直接落在 GPU 上，省掉 CPU 构建和 H2D 搬运
        if isinstance(heads, SparseOneHot):
            heads = heads.dense()
        q = self.bert_encoder(**questions)
        q_embeddings, q_word_h = q.pooler_output, q.last_hidden_state # (bsz, dim_h), (bsz, len, dim_h)
        if triple_batch is None:
            triple_sizes = torch.tensor([triple.shape[0] for triple in triples],
                                        device=heads.device)
            triple_batch = torch.repeat_interleave(
                torch.arange(len(triples), device=heads.device), triple_sizes)
        triples = torch.cat(triples, dim=0)

        e_score = []
        e_score_raw = []
        last_h = torch.zeros_like(q_embeddings)
        for w in range(self.num_ways):
            last_e = heads
            word_attns = []
            rel_probs = []
            ent_probs = []
            ent_probs_raw = []
            for t in range(self.num_steps):
                cq_t = self.step_encoders['way_{}_step_{}'.format(w, t)](
                    torch.cat((q_embeddings, last_h), dim=1) # consider history
                ) # [bsz, dim_h]
                q_logits = torch.sum(cq_t.unsqueeze(1) * q_word_h, dim=2) # [bsz, max_q]
                q_dist = torch.softmax(q_logits, 1) # [bsz, max_q]
                q_dist = q_dist * questions['attention_mask'].float()
                q_dist = q_dist / (torch.sum(q_dist, dim=1, keepdim=True) + 1e-6) # [bsz, max_q]
                word_attns.append(q_dist)
                ctx_h = (q_dist.unsqueeze(1) @ q_word_h).squeeze(1) # [bsz, dim_h]
                ctx_h = ctx_h + cq_t
                last_h = ctx_h

                rel_logit = self.rel_classifiers['way_{}'.format(w)](self.dropout(ctx_h)) # [bsz, num_relations]
                # rel_dist = torch.softmax(rel_logit, 1) # bad
                rel_dist = torch.sigmoid(rel_logit)
                rel_probs.append(rel_dist)

                prev_e = last_e
                last_e = propagate_triples(
                    last_e, rel_dist, triples, triple_batch, self.num_ents)
                if self.use_stay_gate:
                    # 1 跳可达的答案(实测占 44.8%)在第 2 步会被强制传播走，
                    # 这里让模型自己决定留下多少，等价于软化的 self-loop。
                    stay = torch.sigmoid(self.stay_gates['way_{}'.format(w)](ctx_h))
                    last_e = stay * prev_e + (1 - stay) * last_e

                # 钳位会把所有 >1 的分数压成精确的 1.0,实测 12.2% 的样本因此在
                # 最高分上并列,argmax 退化成按实体 id 取第一个。钳位前的分数是有序的
                # (汇入路径越多分越高),留一份出来专门用于推理时打破并列。
                if not self.training:
                    ent_probs_raw.append(last_e)

                if self.score_norm == 'row':
                    # 元素级钳位只要求分数顶过 1,不要求答案排在其他实体之前,
                    # 15.5% 的样本因此在最高分并列(命中率 0.498,明显低于
                    # 最高分唯一的样本)。按行最大值缩放同样压回 [0,1],
                    # 但保留实体间的相对序,迫使模型学排序而不是学饱和。
                    z = last_e.max(dim=1, keepdim=True).values.clamp(min=1).detach()
                    last_e = last_e / z
                else:
                    # reshape >1 scores to 1 in a differentiable way
                    m = last_e.gt(1).float()
                    z = (m * last_e + (1-m)).detach()
                    last_e = last_e / z

                ent_probs.append(last_e)

            hop_res = torch.stack(ent_probs, dim=1) # [bsz, num_hop, num_ent]
            hop_logit = self.hop_selectors['way_{}'.format(w)](q_embeddings)
            hop_attn = torch.softmax(hop_logit, dim=1).unsqueeze(2) # [bsz, num_hop, 1]
            last_e = torch.sum(hop_res * hop_attn, dim=1) # [bsz, num_ent]

            e_score.append(last_e)
            if not self.training:
                # 与 e_score 同样的 hop 加权,只是用未钳位的分数
                e_score_raw.append(
                    torch.sum(torch.stack(ent_probs_raw, dim=1) * hop_attn, dim=1))

        e_score = torch.prod(torch.stack(e_score), dim=0)

        if not self.training:
            return {
                'e_score': e_score,
                'e_score_raw': torch.prod(torch.stack(e_score_raw), dim=0),
                'word_attns': word_attns,
                'rel_probs': rel_probs,
                'ent_probs': ent_probs,
                'hop_attn': hop_attn.squeeze(2)
            }
        else:
            # 推理路径不碰 answers/entity_range，这两份稠密矩阵只在算 loss 时才需要
            if isinstance(answers, SparseOneHot):
                answers = answers.dense()
            if isinstance(entity_range, SparseOneHot):
                entity_range = entity_range.dense()
            weight = answers * self.pos_weight + 1
            loss = torch.sum(entity_range * weight * torch.pow(last_e - answers, 2)) / torch.sum(entity_range * weight)

            return {'loss': loss}
