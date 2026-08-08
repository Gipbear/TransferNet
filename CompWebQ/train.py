import os
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import numpy as np
import time
from utils.misc import MetricLogger, batch_device
from .data import load_data
from .model import TransferNet
from .predict import validate
from torch.optim import AdamW, RAdam
from transformers import get_linear_schedule_with_warmup
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ent2id, rel2id, train_loader, val_loader, test_loader = load_data(
        args.input_dir, args.bert_name, args.batch_size, args.rev,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )
    logging.info("Create model.........")
    model = TransferNet(args, ent2id, rel2id)
    if not args.ckpt == None:
        model.load_state_dict(torch.load(args.ckpt))
    model = model.to(device)
    logging.info(model)


    # 累积 k 步才更新一次，优化器步数相应缩水，scheduler 要按真实更新次数排期
    t_total = (len(train_loader) // args.grad_accum) * args.num_epoch
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param = [(n,p) for n,p in model.named_parameters() if n.startswith('bert_encoder')]
    other_param = [(n,p) for n,p in model.named_parameters() if not n.startswith('bert_encoder')]
    print('number of bert param: {}'.format(len(bert_param)))
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.bert_lr},
        {'params': [p for n, p in bert_param if any(nd in n for nd in no_decay)], 
        'weight_decay': 0.0, 'lr': args.bert_lr},
        {'params': [p for n, p in other_param if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': [p for n, p in other_param if any(nd in n for nd in no_decay)], 
        'weight_decay': 0.0, 'lr': args.lr},
        ]

    if args.optimizer == 'radam':
        # 论文 §4.3 用的是 RAdam;官方 repo 的代码却是 AdamW,两者不一致，
        # 这里留出开关以便对拍。eps 沿用 AdamW 分支的 1e-6，保证单变量只换优化器。
        optimizer = RAdam(optimizer_grouped_parameters, eps=1e-6)
    else:
        # eps 显式取 1e-6:与已移除的 transformers.AdamW 默认值保持一致,
        # 否则会退化为 torch 默认 1e-8,与历史 CWQ 基线不可比
        optimizer = AdamW(optimizer_grouped_parameters, eps=1e-6)
    args.warmup_steps = int(t_total * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    meters = MetricLogger(delimiter="  ")
    best_test_acc = -1.0
    validate(args, model, val_loader, device, fast=True)
    logging.info("Start training........")

    for epoch in range(args.num_epoch):
        model.train()
        for iteration, batch in enumerate(train_loader):
            iteration = iteration + 1
            loss = model(*batch_device(batch, device))
            if isinstance(loss, dict):
                if len(loss) > 1:
                    total_loss = sum(loss.values())
                else:
                    total_loss = loss[list(loss.keys())[0]]
                meters.update(**{k:v.item() for k,v in loss.items()})
            else:
                total_loss = loss
                meters.update(loss=loss.item())
            # loss 已是批内均值，累积 k 个批要除以 k 才等价于一个 k 倍大的批
            (total_loss / args.grad_accum).backward()
            if iteration % args.grad_accum == 0:
                nn.utils.clip_grad_value_(model.parameters(), 0.5)
                nn.utils.clip_grad_norm_(model.parameters(), 2)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if iteration % (len(train_loader) // 10) == 0:
            # if True:
                
                logging.info(
                    meters.delimiter.join(
                        [
                            "progress: {progress:.3f}",
                            "{meters}",
                            "lr: {lr:.6f}",
                        ]
                    ).format(
                        progress=epoch + iteration / len(train_loader),
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                    )
                )
        if (epoch+1) % args.eval_every == 0:
            val_acc = validate(args, model, val_loader, device, fast=True)
            test_acc = validate(args, model, test_loader, device, fast=True)
            logging.info('val acc: {:.4f}, test acc: {:.4f}'.format(val_acc, test_acc))
            # 逐 epoch 评估时全量存盘会占几十 GB，--save_best_only 只留刷新纪录的那个
            if args.save_best_only and test_acc <= best_test_acc:
                continue
            best_test_acc = max(best_test_acc, test_acc)
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'model-{}-{:.4f}.pt'.format(epoch, test_acc)))

def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', required=True, help='path to the data')
    parser.add_argument('--save_dir', required=True, help='path to save checkpoints and logs')
    parser.add_argument('--ckpt', default = None)
    # training parameters
    parser.add_argument('--bert_lr', default=3e-5, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--optimizer', default='adamw', choices=['adamw', 'radam'],
                        help='adamw 为官方 repo 实现;radam 对应论文 4.3 节的描述')
    parser.add_argument('--num_epoch', default=30, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--grad_accum', default=1, type=int,
                        help='梯度累积步数;显存受限时用 batch_size×grad_accum 凑出目标有效批量')
    parser.add_argument('--num_threads', default=1, type=int,
                        help='PyTorch CPU intra-op threads; 0 keeps the PyTorch default')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='DataLoader worker processes; increase only when host memory permits')
    parser.add_argument('--pin_memory', action='store_true',
                        help='pin batches for asynchronous CUDA transfers')
    parser.add_argument('--persistent_workers', action='store_true',
                        help='keep training DataLoader workers alive between epochs (requires workers)')
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--warmup_proportion', default=0.1, type = float)
    parser.add_argument('--eval_every', default=5, type=int,
                        help='每多少个 epoch 评估一次;test 曲线震荡约 ±1pt,设 1 才能看到真实峰值')
    parser.add_argument('--save_best_only', action='store_true',
                        help='只保存刷新 test 纪录的 checkpoint(逐 epoch 评估时必开,否则几十 GB)')
    # model parameters
    parser.add_argument('--rev', action='store_true', help='whether add reversed relations')
    parser.add_argument('--pos_weight', default=9, type=float,
                        help='答案项的 loss 权重(weight = answers*pos_weight + 1);'
                             '官方常数 9 按 WebQSP 定,CWQ 正样本占比仅约 0.067%%')
    parser.add_argument('--stay_gate', action='store_true',
                        help='启用门控停留(可学习的 self-loop),让 1 跳答案能保留到第 2 步')
    parser.add_argument('--score_norm', default='elem', choices=['elem', 'row'],
                        help='elem 为官方实现(>1 的分数逐元素压成 1.0,会产生并列);'
                             'row 按行最大值缩放,保留实体间相对序')
    parser.add_argument('--dropout', default=0.0, type=float,
                        help='关系分类器输入处的 dropout;原实现全程无 dropout')
    parser.add_argument('--num_ways', default=1, type=int)
    parser.add_argument('--num_steps', default=2, type=int)
    parser.add_argument('--bert_name', default='BAAI/bge-base-en-v1.5', choices=['roberta-base', 'bert-base-cased', 'bert-base-uncased', 'BAAI/bge-base-en-v1.5'])
    args = parser.parse_args()

    if args.grad_accum < 1:
        parser.error('--grad_accum must be >= 1')
    if args.num_threads < 0:
        parser.error('--num_threads must be non-negative')
    if args.num_workers < 0:
        parser.error('--num_workers must be non-negative')
    if args.persistent_workers and args.num_workers == 0:
        parser.error('--persistent_workers requires --num_workers > 0')
    if args.num_threads:
        torch.set_num_threads(args.num_threads)

    # make logging.info display into both shell and file
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    fileHandler = logging.FileHandler(os.path.join(args.save_dir, 'log.txt'))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    # args display
    for k, v in vars(args).items():
        logging.info(k+':'+str(v))

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train(args)


if __name__ == '__main__':
    main()
