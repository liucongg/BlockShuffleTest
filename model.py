# -*- coding:utf-8 -*-
# @project: BlockShuffleTest
# @filename: train
# @author: swift
# @source: https://github.com/liucongg/BlockShuffleTest
"""
    文件说明:
            
"""
from transformers import BertModel, BertPreTrainedModel
import torch
import torch.nn as nn


class SentimentAnalysisModel(nn.Module):
    def __init__(self, num_labels=6):
        super(SentimentAnalysisModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.hidden2label = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask, label=None):
        pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[1]
        logits = self.hidden2label(pooled_output)
        predict_label = torch.argmax(logits, dim=-1)
        outputs = (predict_label, logits,)
        if label is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, label)
            outputs = (loss,) + outputs
        return outputs  # (loss), predict_label, logits
