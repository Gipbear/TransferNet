"""训练 Prime 三跳问答上的 TransferNet。"""

from __future__ import annotations

import argparse
import logging
import os
import time

import numpy as np
import torch

from PharmKG.train import train
from Prime.data import load_data

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(message)s"


def build_parser() -> argparse.ArgumentParser:
    """构建 Prime TransferNet 训练参数解析器。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--bert_lr", default=3e-5, type=float)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--num_epoch", default=30, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--save_every", default=5, type=int)
    parser.add_argument("--num_steps", default=3, type=int)
    parser.add_argument("--validate_paths", action="store_true")
    parser.add_argument("--seed", default=17, type=int)
    parser.add_argument("--opt", default="radam")
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--bert_name", default="BAAI/bge-base-en-v1.5")
    return parser


def _configure_logging(save_dir: str, args: argparse.Namespace) -> None:
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    args.log_name = f"{timestamp}_{args.opt}_{args.lr}_{args.batch_size}.log"
    handler = logging.FileHandler(os.path.join(save_dir, args.log_name))
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(handler)
    for key, value in vars(args).items():
        logging.info("%s:%s", key, value)


def main() -> None:
    args = build_parser().parse_args()
    _configure_logging(args.save_dir, args)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args, data_loader=load_data)


if __name__ == "__main__":
    main()
