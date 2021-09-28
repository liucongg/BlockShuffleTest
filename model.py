# -*- coding:utf-8 -*-
# @project: BlockShuffleTest
# @filename: model
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2021/9/27 10:52
"""
    文件说明:
            
"""
from transformers import BertModel, BertPreTrainedModel
import torch
import torch.nn as nn


class SentimentAnalysisModel(BertPreTrainedModel):
    def __init__(self, config):
        super(SentimentAnalysisModel, self).__init__(config)
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask, label=None):
        pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[1]
        logits = self.classifier(pooled_output)
        predict_label = torch.argmax(logits, dim=-1)
        outputs = (predict_label, logits,)
        if label is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, label)
            outputs = (loss,) + outputs
        return outputs  # (loss), predict_label, logits
