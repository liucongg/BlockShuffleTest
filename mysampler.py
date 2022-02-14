# -*- coding:utf-8 -*-
# @project: BlockShuffleTest
# @filename: train
# @author: swift
# @source: https://github.com/liucongg/BlockShuffleTest

import math
import random
from torch.utils.data import Sampler
import time


class MySampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last=False):
        T1 = time.time()
        sorted_arr = sorted(enumerate(data_source), key=lambda x: len(x[1]["input_ids"]), reverse=True)
        self.indices = [i[0] for i in sorted_arr]
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.chunks = self.get_chunks()
    
    def get_chunks(self):    # split indices of data_source into chunks composed of batch
        chunks = []
        for i in range(0, len(self.indices), self.batch_size):
            batch = self.indices[i:i+self.batch_size]
            chunks.append(batch)
        if len(chunks[-1]) != self.batch_size and self.drop_last:
            chunks = chunks[:-1]
        random.shuffle(chunks)
        return chunks

    def __iter__(self):
        for batch in self.chunks:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.indices) // self.batch_size
        else:
            return (len(self.indices) + self.batch_size - 1) // self.batch_size
