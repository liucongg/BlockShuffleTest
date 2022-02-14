# -*- coding:utf-8 -*-
# @project: BlockShuffleTest
# @filename: train
# @author: swift
# @source: https://github.com/liucongg/BlockShuffleTest
"""
    文件说明:
            
"""
import torch
import os
import random
import numpy as np
import argparse
import logging
from transformers import BertTokenizer
from data_set import SentimentAnalysisDataSet, collate_func_sentiment_analysis
from mysampler import MySampler
from model import SentimentAnalysisModel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
from sklearn.metrics import f1_score, accuracy_score
import json
import time

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)



def evaluate(model, device, dev_data, args):
    test_sampler = SequentialSampler(dev_data)
    test_data_loader = DataLoader(dev_data, sampler=test_sampler,
                                  batch_size=args.test_batch_size, collate_fn=collate_func_sentiment_analysis)
    iter_bar = tqdm(test_data_loader, desc="iter", disable=False)
    true_label = []
    pre_label = []
    model.eval()
    for step, batch in enumerate(iter_bar):
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)
            predict_label, _ = model.forward(input_ids, attention_mask)
            true_label.extend(label.cpu().numpy())
            pre_label.extend(predict_label.cpu().numpy())
    true_label = np.array(true_label)
    pre_label = np.array(pre_label)
    f1 = f1_score(true_label, pre_label, average='macro')
    acc = accuracy_score(true_label, pre_label)
    return acc, f1


def train_ori_time(model, device, tokenizer, args):
    train_batch_size = int(args.train_batch_size)
    train_data = SentimentAnalysisDataSet(tokenizer, args.max_len, args.data_dir, "train", args.train_file_path)
    train_sampler = RandomSampler(train_data)
    train_data_loader = DataLoader(train_data, sampler=train_sampler,
                                   batch_size=train_batch_size, collate_fn=collate_func_sentiment_analysis)
    total_steps = int(len(train_data_loader) * args.num_train_epochs)
    logger.info("总训练步数为:{}".format(total_steps))
    model.to(device)
    no_decay = ["bias", "LayerNorm.weight"]
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 
         'lr': args.learning_rate},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
        'weight_decay': 0.0,
         'lr': args.learning_rate}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()
    torch.cuda.empty_cache()
    model.train()
    T1 = time.time()
    for iepoch in trange(0, args.num_train_epochs, desc="Epoch", disable=False):
        iter_bar = tqdm(train_data_loader, desc="Iter", disable=False)
        for step, batch in enumerate(iter_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)

            _, logits = model.forward(input_ids, attention_mask)
            loss = criterion(logits, label)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    T2 = time.time()
    print("原始DataLoader，运行2个epoch时间为{}秒".format(T2 - T1))
    dev_data = SentimentAnalysisDataSet(tokenizer, args.max_len, args.data_dir, "dev", args.dev_file_path)
    acc, f1 = evaluate(model, device, dev_data, args)
    print("origin train acc: {} f1: {}".format(acc, f1))


def train_block_shuffle_time(model, device, tokenizer, args):
    train_batch_size = int(args.train_batch_size)
    train_data = SentimentAnalysisDataSet(tokenizer, args.max_len, args.data_dir, "train", args.train_file_path)
    train_data_sampler = MySampler(train_data, train_batch_size)
    train_data_loader = DataLoader(train_data, batch_sampler=train_data_sampler, collate_fn=collate_func_sentiment_analysis)

    total_steps = int(len(train_data_loader) * args.num_train_epochs)
    logger.info("总训练步数为:{}".format(total_steps))
    model.to(device)
    no_decay = ["bias", "LayerNorm.weight"]
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.learning_rate}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                num_training_steps=total_steps)
    torch.cuda.empty_cache()
    model.train()
    T1 = time.time()
    for iepoch in trange(0, args.num_train_epochs, desc="Epoch", disable=False):
        iter_bar = tqdm(train_data_loader, desc="Iter", disable=False)
        for step, batch in enumerate(iter_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)

            outputs = model.forward(input_ids, attention_mask, label)
            loss = outputs[0]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    T2 = time.time()
    print("BlockShuffleDataLoader，运行2个epoch时间为{}秒".format(T2 - T1))
    dev_data = SentimentAnalysisDataSet(tokenizer, args.max_len, args.data_dir, "dev", args.dev_file_path)
    acc, f1 = evaluate(model, device, dev_data, args)
    print("shuffle train acc: {} f1: {}".format(acc, f1))


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=0, type=int, help='gpu device id')
    parser.add_argument('--train_file_path', default='data/train.json', type=str, help='')
    parser.add_argument('--dev_file_path', default='data/test.json', type=str, help='')
    parser.add_argument('--data_dir', default='data/', type=str, help='')
    parser.add_argument('--num_train_epochs', default=2, type=int, help='')
    parser.add_argument('--train_batch_size', default=4, type=int, help='')
    parser.add_argument('--test_batch_size', default=4, type=int, help='')
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='')
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="")
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='')
    parser.add_argument('--save_model_steps', default=12, type=int, help='')
    parser.add_argument('--logging_steps', default=5, type=int, help='')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='')
    parser.add_argument('--output_dir', default='output_dir', type=str,
                        help='')
    parser.add_argument('--is_block_shuffle', type=bool, default=True, help='')
    parser.add_argument('--seed', type=int, default=2020, help='')
    parser.add_argument('--max_len', type=int, default=256, help='')
    parser.add_argument('--num_labels', type=int, default=6, help='')
    return parser.parse_args()


def main():
    args = set_args()
    device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")
    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    model = SentimentAnalysisModel(num_labels=args.num_labels)
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    train_ori_time(model, device, tokenizer, args)
    train_block_shuffle_time(model, device, tokenizer, args)


if __name__ == '__main__':
    main()
