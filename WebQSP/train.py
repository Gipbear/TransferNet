import os
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import numpy as np
import time
from tqdm import tqdm
from utils.misc import MetricLogger, batch_device, RAdam
from utils.lr_scheduler import get_linear_schedule_with_warmup
from utils.path_utils import build_valid_edges_dict
from .data import load_data
from .model import TransferNet
from .predict import id_set, validate
from torch.amp import autocast, GradScaler
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()

torch.set_num_threads(1) # avoid using multiple cpus


def validate_qa_acc(model, data, device):
    model.eval()
    count = 0
    correct = 0
    hop_count = {0: [], 1: []}
    with torch.no_grad():
        for batch in tqdm(data, total=len(data), desc='validate Hit@1', dynamic_ncols=True, mininterval=1.0):
            outputs = model(*batch_device(batch, device), return_intermediates=False)
            e_score = outputs['e_score'].detach().cpu()
            top1_idx = e_score.argmax(dim=1).tolist()
            hop_attn = outputs['hop_attn'].detach().cpu()
            for i in range(len(batch[2])):
                gold_ids = id_set(batch[2][i])
                hit = bool(gold_ids) and (top1_idx[i] in gold_ids)
                hop = int(hop_attn[i].argmax().item())
                hop_count.setdefault(hop, []).append(float(hit))
                count += 1
                correct += int(hit)
            del outputs, e_score, hop_attn
    acc = correct / count if count else 0.0
    logging.info(
        "Hit@1: %.4f  pred hop accuracy: 1-hop %.4f (total %d), 2-hop %.4f (total %d)",
        acc,
        sum(hop_count.get(0, [])) / (len(hop_count.get(0, [])) + 0.1),
        len(hop_count.get(0, [])),
        sum(hop_count.get(1, [])) / (len(hop_count.get(1, [])) + 0.1),
        len(hop_count.get(1, [])),
    )
    return acc


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    ent2id, rel2id, triples, train_loader, val_loader = load_data(
        args.input_dir,
        args.bert_name,
        args.batch_size,
    )

    triples_list = None
    valid_edges_dict = None
    if args.validate_paths:
        triples_list = [[int(s), int(r), int(o)] for s, r, o in triples.tolist()]
        valid_edges_dict = build_valid_edges_dict(triples_list)

    logging.info("Create model.........")
    model = TransferNet(args, ent2id, rel2id, triples)
    if not args.ckpt == None:
        model.load_state_dict(torch.load(args.ckpt))
    model = model.to(device)
    # model.triples = model.triples.to(device)
    model.Msubj = model.Msubj.to(device)
    model.Mobj = model.Mobj.to(device)
    model.Mrel = model.Mrel.to(device)
    logging.info(model)


    t_total = len(train_loader) * args.num_epoch
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
    # optimizer_grouped_parameters = [{'params':model.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr}]
    if args.opt == 'adam':
        optimizer = optim.Adam(optimizer_grouped_parameters)
    elif args.opt == 'radam':
        optimizer = RAdam(optimizer_grouped_parameters)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(optimizer_grouped_parameters)
    else:
        raise NotImplementedError
    args.warmup_steps = int(t_total * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    scaler = GradScaler('cuda', enabled=torch.cuda.is_available())
    meters = MetricLogger(delimiter="  ")
    # validate(args, model, val_loader, device)
    logging.info("Start training........")

    for epoch in range(args.num_epoch):
        model.train()
        for iteration, batch in enumerate(train_loader):
            iteration = iteration + 1
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=torch.cuda.is_available()):
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
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

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
        if (epoch+1)%5 == 0:
            if args.validate_paths:
                acc = validate(args, model, val_loader, triples_list, valid_edges_dict, device)
            else:
                acc = validate_qa_acc(model, val_loader, device)
            logging.info(acc)
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'model-{}-{:.4f}.pt'.format(epoch, acc)))

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
    parser.add_argument('--num_epoch', default=50, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--validate_paths', action='store_true',
                        help='Run full MMR path validation during training. Uses much more CPU memory.')
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--opt', default='radam', type = str)
    parser.add_argument('--warmup_proportion', default=0.1, type = float)
    # model parameters
    parser.add_argument('--bert_name', default='BAAI/bge-base-en-v1.5')
    args = parser.parse_args()

    # make logging.info display into both shell and file
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    args.log_name = time_ + '_{}_{}_{}.log'.format(args.opt, args.lr, args.batch_size)
    fileHandler = logging.FileHandler(os.path.join(args.save_dir, args.log_name))
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
