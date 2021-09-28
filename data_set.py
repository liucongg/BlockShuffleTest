# -*- coding:utf-8 -*-
# @project: TestCode
# @filename: BlockShuffleTest
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2021/9/27 10:46
"""
    文件说明:
            
"""
import torch
import json
import os
from torch.utils.data import Dataset
import logging
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


class SentimentAnalysisDataSet(Dataset):
    def __init__(self, tokenizer, max_len, data_dir, data_set_name, path_file=None, is_overwrite=False):
        super(SentimentAnalysisDataSet, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = {'angry': 0, 'happy': 1, 'neutral': 2, 'surprise': 3, 'sad': 4, 'fear': 5}
        self.id2label = {0: "angry", 1: "happy", 2: "neutral", 3: "surprise", 4: "sad", 5: "fear"}
        cached_feature_file = os.path.join(data_dir, "cached_{}_{}".format(data_set_name, max_len))
        if os.path.exists(cached_feature_file) and not is_overwrite:
            logger.info("已经存在缓存文件{}，直接加载".format(cached_feature_file))
            self.data_set = torch.load(cached_feature_file)["data_set"]
        else:
            logger.info("不存在缓存文件{}，进行数据预处理操作".format(cached_feature_file))
            self.data_set = self.load_data(path_file)
            logger.info("数据预处理操作完成，将处理后的数据存到{}中，作为缓存文件".format(cached_feature_file))
            torch.save({"data_set": self.data_set}, cached_feature_file)

    def load_data(self, path_file):
        data_set = []
        with open(path_file, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                sample = json.loads(line.strip())
                input_ids, attention_mask, label = self.convert_featrue(sample)
                sample["input_ids"] = input_ids
                sample["attention_mask"] = attention_mask
                sample["label"] = label
                data_set.append(sample)
        return data_set

    def convert_featrue(self, sample):
        label = self.label2id[sample["label"]]
        tokens = self.tokenizer.tokenize(sample["text"])
        if len(tokens) > self.max_len - 2:
            tokens = tokens[:self.max_len - 2]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(attention_mask)
        return input_ids, attention_mask, label

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        instance = self.data_set[idx]
        return instance


def collate_func_sentiment_analysis(batch_data):
    batch_size = len(batch_data)
    if batch_size == 0:
        return {}
    input_ids_list, attention_mask_list, labels_list = [], [], []
    for instance in batch_data:
        input_ids_temp = instance["input_ids"]
        attention_mask_temp = instance["attention_mask"]
        labels_temp = instance["label"]
        input_ids_list.append(torch.tensor(input_ids_temp, dtype=torch.long))
        attention_mask_list.append(torch.tensor(attention_mask_temp, dtype=torch.long))
        labels_list.append(labels_temp)
    return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=0),
            "attention_mask": pad_sequence(attention_mask_list, batch_first=True, padding_value=0),
            "label": torch.tensor(labels_list, dtype=torch.long)}
