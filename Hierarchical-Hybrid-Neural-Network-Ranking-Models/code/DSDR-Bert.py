#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')

import random
import logging
import os
import copy
import time
import math 
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    cohen_kappa_score,
)

from transformers import (
    BertModel,
    BertTokenizer,
    AdamW,
    get_scheduler
)
from torch.utils.data import DataLoader, Dataset


# 模型加载选择GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))


# set random values
def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
        
# prepare logger
def get_logger():
    current_time = time.strftime("%Y_%m_%d_%H_%M_%S")
    os.makedirs('logs', exist_ok=True)
    log_path = os.path.join('logs', f'{current_time}_train.log')
    logging.basicConfig(
        filename=log_path,
        format="%(asctime)s | %(funcName)10s | %(message)s",
        datefmt="%Y-%m-%d-%H:%M:%S",
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    return logger

# create dataset
class LingFeatDataset(Dataset):
    def __init__(self, df):
        self.df = df
        
    def num_class(self):
        # Return number of classes
        return len(self.df['label'].unique())
        
    def __len__(self):
        # Number of rows
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        # Retreive 'Text', 'Label', 'Index' from dataframe
        source = self.df['sentence'].values[idx]
        target = self.df['label'].values[idx]
        item_idx = self.df.index[idx]
        return {
            'source': source,
            'target': target,
            'item_idx': item_idx
        }

# collate function
class LingFeatBatchGenerator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
        sources = [item['source'] for item in batch]
        targets = [item['target'] for item in batch]
        item_idxs = [item['item_idx'] for item in batch]    # map item during inference stage

         # 将句子编码为 BERT 的输入格式
        encoding = self.tokenizer(
            sources,
            padding='max_length',
            truncation=True,
            max_length=64,
            return_tensors='pt'
        )
        labels = torch.LongTensor(targets)
        
        return encoding, labels, item_idxs


def _optimizer(model):
    # prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.02},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=3e-5
    )
    return optimizer

def _scheduler(optimizer, total_steps):
    # prepare scheduler
    scheduler = get_scheduler(
        'linear',
        optimizer=optimizer,
        num_warmup_steps=(total_steps * 0.1),
        num_training_steps=total_steps
    )
    return scheduler

## 定义一个对数据集训练一轮的函数
def bert_train_epoch(model, iterator, optimizer, scheduler, criterion):
    epoch_loss = 0;epoch_acc = 0
    train_corrects = 0;train_num = 0
    model.train()
    for batch_idx, batch_item in enumerate(iterator):
        inputs, labels, _ = batch_item
        input_ids, attention_masks = inputs['input_ids'], inputs['attention_mask']
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        labels = labels.to(device)    # normal tensor no in-place operation
        
        output = model(input_ids, attention_masks)
        loss = criterion(output, labels)
        loss.backward()
        
        pre_lab = torch.argmax(output, 1)
        train_corrects += torch.sum(pre_lab == labels)
        train_num += len(labels) ## 样本数量
        epoch_loss += loss.item()
        
        # TODO: apply gradient accumulation
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    ## 所有样本的平均损失和精度
    epoch_loss = epoch_loss / train_num
    epoch_acc = train_corrects.double().item() / train_num
    return epoch_loss, epoch_acc


class BERTDifficultyClassifier(nn.Module):
    def __init__(self, bert_model_name, num_labels):
        super(BERTDifficultyClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)  # 使用BERT
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)  # 分类层

    def forward(self, input_ids, attention_mask):
        # BERT 输出 [CLS] token 的向量
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # 获取 [CLS] 位置的输出
        
        # 分类器输出难度标签概率
        logits = self.classifier(cls_output)
        return logits



# model_name = 'bert'
pretrained_model_name = r"bert-base-uncased"
tokenizer_class = BertTokenizer



# load tokenizer
# model will be loaded for each kth-fold
tokenizer = tokenizer_class.from_pretrained(pretrained_model_name)

# init collate_fn
batch_generator = LingFeatBatchGenerator(tokenizer)


final_sentence_level = pd.read_csv(r'final sentence level/sentence corpus.csv')


path = r"data collect\data_split.csv"
data_df = pd.read_csv(path)
labelMap = {'Ele_Txt' : 0, 'Int_Txt' : 1, 'Adv_Txt' : 2}
data_df['labelcode'] = data_df['label'].map(labelMap)
data_df['index'] = range(len(data_df))
# 将数据集划分为 训练集， 验证机 和 测试集
train_df, test_df =  train_test_split(data_df, test_size=0.2, random_state=2023)
print(len(train_df))
print(len(test_df))


# 在句子数据集中筛选出属于训练集和测试集的句子
train_sentence_df = final_sentence_level[final_sentence_level['index'].isin(train_df['index'])]


# load logger
logger = get_logger()
batch_size = 16
epochs = 10
time_list = ['First time', 'Second time', 'Third time']
logger.info('********** DSDR-Bert Start Run **********')
for i in range(3):
    # set seed for reproducibility
    set_seed(2023+i)
    logger.info(f'********** {time_list[i]} **********')
    
    train_dataset = LingFeatDataset(train_sentence_df)
    train_loader = DataLoader(train_dataset, collate_fn=batch_generator, batch_size=batch_size, shuffle=True)
    
    total_steps = len(train_loader) * epochs    # TODO: divide by accumulation steps
    model = BERTDifficultyClassifier(pretrained_model_name, train_dataset.num_class())    # need to load model for every k fold
    model.to(device)
    optimizer = _optimizer(model)
    scheduler = _scheduler(optimizer, total_steps)
    criterion = nn.CrossEntropyLoss()
    
    ## 使用训练集训练模型，验证集测试模型
    for epoch in range(epochs):
        start_time = time.time()
        train_loss, train_acc = bert_train_epoch(model, train_loader, optimizer, scheduler, criterion)
        end_time = time.time()
        logger.info('TRAIN SET    | Epoch: {:.3f} | Loss: {:.5f} | Accuracy: {:.4f}'.format(
            epoch+1,
            train_loss,
            train_acc
        ))
        print("Epoch:" ,epoch+1 ,"|" ,"Epoch Time: ",end_time - start_time, "s")
        print("Train Loss:", train_loss, "|" ,"Train Acc: ",train_acc)
        
    # 保存模型参数到文件
    torch.save(model.state_dict(), r'save model\BModel.pth')