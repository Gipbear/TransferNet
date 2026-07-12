import torch
import torch.nn as nn
from transformers import AutoModel
from utils.huggingface import from_pretrained_local_first

class TransferNet(nn.Module):
    def __init__(self, args, ent2id, rel2id, triples):
        super().__init__()
        self.args = args
        self.num_steps = 2
        num_relations = len(rel2id)
        # self.triples = triples

        Tsize = len(triples)
        Esize = len(ent2id)
        idx = torch.LongTensor([i for i in range(Tsize)])
        self.Msubj = torch.sparse_coo_tensor(
            torch.stack((idx, triples[:,0])), torch.FloatTensor([1] * Tsize), torch.Size([Tsize, Esize]))
        self.Mobj = torch.sparse_coo_tensor(
            torch.stack((idx, triples[:,2])), torch.FloatTensor([1] * Tsize), torch.Size([Tsize, Esize]))
        self.Mrel = torch.sparse_coo_tensor(
            torch.stack((idx, triples[:,1])), torch.FloatTensor([1] * Tsize), torch.Size([Tsize, num_relations]))
        print('triple size: {}'.format(Tsize))

        self.bert_encoder = from_pretrained_local_first(
            AutoModel, args.bert_name, return_dict=True
        )
        dim_hidden = self.bert_encoder.config.hidden_size

        self.step_encoders = []
        for i in range(self.num_steps):
            m = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.Tanh()
            )
            self.step_encoders.append(m)
            self.add_module('step_encoders_{}'.format(i), m)

        self.rel_classifier = nn.Linear(dim_hidden, num_relations)

        self.hop_selector = nn.Linear(dim_hidden, self.num_steps)

    def _one_hot_from_index_lists(self, index_lists, device):
        heads = torch.zeros(
            (len(index_lists), self.Msubj.shape[1]),
            device=device,
            dtype=torch.float32,
        )
        for row, ids in enumerate(index_lists):
            ids = ids.to(device=device, dtype=torch.long)
            if ids.numel() > 0:
                heads[row].scatter_(0, ids, 1.0)
        return heads

    def _sparse_loss(self, scores, answers, entity_range):
        numerator = scores.new_tensor(0.0)
        denominator = scores.new_tensor(0.0)
        for row, range_ids in enumerate(entity_range):
            range_ids = range_ids.to(device=scores.device, dtype=torch.long)
            if range_ids.numel() == 0:
                continue
            answer_ids = answers[row].to(device=scores.device, dtype=torch.long)
            target = torch.isin(range_ids, answer_ids).to(dtype=scores.dtype)
            weight = target * 99 + 1
            row_scores = scores[row].index_select(0, range_ids)
            numerator = numerator + torch.sum(weight * torch.pow(row_scores - target, 2))
            denominator = denominator + torch.sum(weight)
        return numerator / denominator.clamp_min(1.0)

    def follow(self, e, r):
        # CUDA sparse mm doesn't support float16. Casting with .float() alone
        # isn't enough under AMP: autocast can silently re-cast fp32 inputs
        # back to fp16 when entering ops outside its safelist. Disable autocast
        # explicitly for this region.
        with torch.amp.autocast('cuda', enabled=False):
            e32 = e.float()
            r32 = r.float()
            x = torch.sparse.mm(self.Msubj, e32.t()) * torch.sparse.mm(self.Mrel, r32.t())
            out = torch.sparse.mm(self.Mobj.t(), x).t()
        return out # [bsz, Esize], fp32

    def forward(self, heads, questions, answers=None, entity_range=None, return_intermediates=None):
        if isinstance(heads, list):
            heads = self._one_hot_from_index_lists(heads, questions['input_ids'].device)

        collect_intermediates = (not self.training) if return_intermediates is None else return_intermediates
        q = self.bert_encoder(**questions)
        q_embeddings, q_word_h = q.pooler_output, q.last_hidden_state # (bsz, dim_h), (bsz, len, dim_h)

        last_e = heads
        word_attns = [] if collect_intermediates else None
        rel_probs = [] if collect_intermediates else None
        ent_probs = [] if collect_intermediates else None
        hop_attn = torch.softmax(self.hop_selector(q_embeddings), dim=1)
        weighted_e = None
        for t in range(self.num_steps):
            cq_t = self.step_encoders[t](q_embeddings) # [bsz, dim_h]
            q_logits = torch.sum(cq_t.unsqueeze(1) * q_word_h, dim=2) # [bsz, max_q]
            q_dist = torch.softmax(q_logits, 1) # [bsz, max_q]
            q_dist = q_dist * questions['attention_mask'].float()
            q_dist = q_dist / (torch.sum(q_dist, dim=1, keepdim=True) + 1e-6) # [bsz, max_q]
            if collect_intermediates:
                word_attns.append(q_dist)
            ctx_h = (q_dist.unsqueeze(1) @ q_word_h).squeeze(1) # [bsz, dim_h]

            rel_logit = self.rel_classifier(ctx_h) # [bsz, num_relations]
            # rel_dist = torch.softmax(rel_logit, 1) # bad
            rel_dist = torch.sigmoid(rel_logit)
            if collect_intermediates:
                rel_probs.append(rel_dist)

            # sub, rel, obj = self.triples[:,0], self.triples[:,1], self.triples[:,2]
            # sub_p = last_e[:, sub] # [bsz, #tri]
            # rel_p = rel_dist[:, rel] # [bsz, #tri]
            # obj_p = sub_p * rel_p
            # last_e = torch.index_add(torch.zeros_like(last_e), 1, obj, obj_p)

            last_e = self.follow(last_e, rel_dist) # faster than index_add

            # reshape >1 scores to 1 in a differentiable way
            m = last_e.gt(1).float()
            z = (m * last_e + (1-m)).detach()
            last_e = last_e / z

            weighted_step = last_e * hop_attn[:, t:t + 1]
            weighted_e = weighted_step if weighted_e is None else weighted_e + weighted_step
            if collect_intermediates:
                ent_probs.append(last_e)

        last_e = weighted_e # [bsz, num_ent]

        if not self.training:
            outputs = {
                'e_score': last_e,
                'hop_attn': hop_attn,
            }
            if collect_intermediates:
                outputs.update({
                    'word_attns': word_attns,
                    'rel_probs': rel_probs,
                    'ent_probs': ent_probs,
                })
            return outputs
        else:
            if isinstance(answers, list):
                loss = self._sparse_loss(last_e, answers, entity_range)
            else:
                weight = answers * 99 + 1
                loss = torch.sum(entity_range * weight * torch.pow(last_e - answers, 2)) / torch.sum(entity_range * weight)

            return {'loss': loss}
