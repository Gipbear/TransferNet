import torch
import os
import sys
from collections import defaultdict
from transformers import AutoTokenizer
from tqdm import tqdm
from utils.misc import invert_dict
from utils.huggingface import from_pretrained_local_first


def iter_file_with_progress(path, desc, unit):
    total = os.path.getsize(path)
    with open(path, 'rb') as fp:
        with tqdm(
            total=total,
            desc=desc,
            unit=unit,
            unit_scale=True,
            dynamic_ncols=True,
            mininterval=1.0,
            file=sys.stdout,
        ) as pbar:
            for raw_line in fp:
                pbar.update(len(raw_line))
                yield raw_line.decode('utf-8')


def normalize_answers(answer_text):
    return [answer.strip() for answer in answer_text.split('|') if answer.strip()]


def collate(batch):
    batch = list(zip(*batch))
    topic_entity, question, answer, entity_range = batch
    topic_entity = list(topic_entity)
    question = {k:torch.cat([q[k] for q in question], dim=0) for k in question[0]}
    answer = list(answer)
    entity_range = list(entity_range)
    return topic_entity, question, answer, entity_range


class Dataset(torch.utils.data.Dataset):
    def __init__(self, questions, ent2id):
        self.questions = questions
        self.ent2id = ent2id

    def __getitem__(self, index):
        topic_entity, question, answer, entity_range = self.questions[index]
        return (
            torch.LongTensor(topic_entity),
            question,
            torch.LongTensor(answer),
            torch.LongTensor(entity_range),
        )

    def __len__(self):
        return len(self.questions)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, input_dir, fn, bert_name, ent2id, rel2id, batch_size, training=False):
        print('Reading questions from {}'.format(fn))
        self.tokenizer = from_pretrained_local_first(AutoTokenizer, bert_name)
        self.ent2id = ent2id
        self.rel2id = rel2id
        self.id2ent = invert_dict(ent2id)
        self.id2rel = invert_dict(rel2id)



        sub_map = defaultdict(list)
        train_path = os.path.join(input_dir, 'fbwq_full/train.txt')
        for line in iter_file_with_progress(train_path, desc='build WebQSP adjacency', unit='B'):
            l = line.strip().split('\t')
            s = l[0].strip()
            o = l[2].strip()
            sub_map[ent2id[s]].append(ent2id[o])


        data = []
        qa_text = []
        missing_answer_count = 0
        skipped_missing_answer_count = 0
        missing_answer_examples = []
        for line_no, line in enumerate(iter_file_with_progress(fn, desc='parse WebQSP questions', unit='B'), 1):
            line = line.strip()
            if line == '':
                continue
            line = line.split('\t')
            # if no answer
            if len(line) != 2:
                continue
            question = line[0].split('[')
            question_1 = question[0]
            question_2 = question[1].split(']')
            head = question_2[0].strip()
            question_2 = question_2[1]
            # question = question_1 + 'NE' + question_2
            question = question_1.strip()
            ans = normalize_answers(line[1])

            # if (head, ans[0]) not in so_map:
            #     continue

            valid_ans = []
            for a in ans:
                if a in ent2id:
                    valid_ans.append(ent2id[a])
                else:
                    missing_answer_count += 1
                    if len(missing_answer_examples) < 5:
                        missing_answer_examples.append((line_no, a))
            if not valid_ans:
                skipped_missing_answer_count += 1
                continue

            head = ent2id[head]
            entity_range = set()
            for o in sub_map[head]:
                entity_range.add(o)
                entity_range.update(sub_map[o])
            entity_range = list(entity_range)

            head = [head]
            data.append([head, question, valid_ans, entity_range])
            # 与 data 严格同序记录过滤后存活行的原始问句文本，
            # 供 dump_scores 等下游按 batch 顺序取用，避免与得分错位。
            qa_text.append(question)

        tokenize_batch_size = 512
        for start in tqdm(range(0, len(data), tokenize_batch_size), desc='tokenize WebQSP questions', unit='batch', dynamic_ncols=True, mininterval=1.0, file=sys.stdout):
            end = min(start + tokenize_batch_size, len(data))
            encoded = self.tokenizer(
                [item[1] for item in data[start:end]],
                max_length=64,
                padding='max_length',
                return_tensors="pt",
            )
            for offset, item in enumerate(data[start:end]):
                item[1] = {k: v[offset:offset + 1] for k, v in encoded.items()}

        print('data number: {}'.format(len(data)))
        if missing_answer_count:
            print(
                'Warning: skipped {} unknown answer entities and dropped {} questions with no valid answers while reading {}. Examples: {}'.format(
                    missing_answer_count,
                    skipped_missing_answer_count,
                    fn,
                    missing_answer_examples,
                )
            )
        
        dataset = Dataset(data, ent2id)
        # 过滤后存活行的原始问句文本，顺序与 dataset 一一对应。
        self.qa_text = qa_text

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=training,
            collate_fn=collate,
            pin_memory=torch.cuda.is_available(),
        )


def load_graph(input_dir):
    """加载 WebQSP 图谱，不构造问答 DataLoader。"""
    print('Read data...')
    ent2id = {}
    for line in open(os.path.join(input_dir, 'fbwq_full/entities.dict')):
        l = line.strip().split('\t')
        ent2id[l[0].strip()] = len(ent2id)
    # print(len(ent2id))
    # print(max(ent2id.values()))
    rel2id = {}
    for line in open(os.path.join(input_dir, 'fbwq_full/relations.dict')):
        l = line.strip().split('\t')
        rel2id[l[0].strip()] = int(l[1])

    triples = []
    for line in open(os.path.join(input_dir, 'fbwq_full/train.txt')):
        l = line.strip().split('\t')
        s = ent2id[l[0].strip()]
        p = rel2id[l[1].strip()]
        o = ent2id[l[2].strip()]
        triples.append((s, p, o))
        p_rev = rel2id[l[1].strip()+'_reverse']
        triples.append((o, p_rev, s))
    triples = torch.LongTensor(triples)

    return ent2id, rel2id, triples


def load_data(input_dir, bert_name, batch_size):
    ent2id, rel2id, triples = load_graph(input_dir)

    train_data = DataLoader(input_dir, os.path.join(input_dir, 'QA_data/WebQuestionsSP/qa_train_webqsp.txt'), bert_name, ent2id, rel2id, batch_size, training=True)
    test_data = DataLoader(input_dir, os.path.join(input_dir, 'QA_data/WebQuestionsSP/qa_test_webqsp_fixed.txt'), bert_name, ent2id, rel2id, batch_size)

    return ent2id, rel2id, triples, train_data, test_data
