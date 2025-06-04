#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')

import os
import random
import logging
import math
import time
import pandas as pd
from collections import Counter
from itertools import permutations
import re
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    cohen_kappa_score,
)


# 模型加载选择GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

# set seed values
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

# Get all permutation pairs out of an array
def get_pairs(arr):
    return torch.tensor(list(permutations(arr, 2)))

def text_to_pairs(data, pairs):
    pairsFlat = pairs.reshape(-1)
    data_pairs = data[pairsFlat].split(pairs.size(0), dim=0)
    data_pairs = torch.stack(data_pairs, dim=0).reshape(pairs.size(0), -1)
    return data_pairs


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


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_labels, num_heads=1):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads)
        self.num_labels = num_labels
        
        # 难度表示向量 C (可学习参数)
        self.difficulty_vectors = nn.Parameter(torch.randn(num_labels, hidden_size))
        init.xavier_uniform_(self.difficulty_vectors)  # 初始化

    def forward(self, transformer_output, sentence_mask=None):
        # transformer_output 形状为 (batch_size, num_sentences, hidden_size)
        transformer_output = transformer_output.permute(1, 0, 2)  # (num_sentences, batch_size, hidden_size)
        
        # 将 C 作为 Query，transformer_output 作为 Key 和 Value
        query = self.difficulty_vectors.unsqueeze(1).expand(self.num_labels, transformer_output.size(1), -1)  # (num_labels, batch_size, hidden_size)
        
        # 交叉注意力计算 (query, key, value)，并传入 attention_mask 来 mask 掉无效句子
        R, _ = self.attention(query, transformer_output, transformer_output, key_padding_mask=sentence_mask)  # (num_labels, batch_size, hidden_size)
        
        R = R.permute(1, 0, 2)  # (batch_size, num_labels, hidden_size)
        return R

class TextDifficultyModel(nn.Module):
    def __init__(self, bmodel, num_labels, transformer_layers=3, hidden_size=768, max_len=5000, num_heads=1):
        super(TextDifficultyModel, self).__init__()
        
        # 使用训练好的 BModel
        self.bmodel = bmodel
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model=hidden_size, max_len=max_len)
        
        # Transformer 编码器用于上下文补充
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # 交叉注意力机制
        self.cross_attention = CrossAttention(hidden_size=hidden_size, num_labels=num_labels, num_heads=num_heads)
        
        # 线性分类层
        self.classifier = nn.Linear(hidden_size, 400, bias=True)
        
        # MeanPooling
        self.mean_pooling = nn.AdaptiveAvgPool1d(1)
        
        self.regressor = nn.Sequential(nn.Linear(400 * 2, 400),
                       nn.ReLU(),
                       nn.Linear(400, 1))

    def forward(self, input_ids, attention_masks, pairs):
        batch_size, num_sentences, seq_len = input_ids.size()  # batch_size, m, max_len
        sentence_embeddings = []

        # 逐句处理每个句子，并使用训练好的 BModel 提取表示
        for i in range(num_sentences):
            input_ids_batch = input_ids[:, i, :]  # (batch_size, seq_len)
            attention_mask_batch = attention_masks[:, i, :]  # (batch_size, seq_len)
            
            with torch.no_grad():
                outputs = self.bmodel.bert(input_ids=input_ids_batch, attention_mask=attention_mask_batch)
                cls_output = outputs.last_hidden_state[:, 0, :]  # 获取 [CLS] 位置的输出
                sentence_embeddings.append(cls_output)
        
        # 将所有句子表示组合成 H 序列
        sentence_embeddings = torch.stack(sentence_embeddings, dim=1)  # (batch_size, m, hidden_size)

        # 创建句子级别的 mask，mask 填充的句子 (全零句子的 attention mask 应该全为 0)
        sentence_mask = (attention_masks.sum(dim=2) == 0)  # 计算句子级 mask: (batch_size, m)
        
        # 位置编码
        sentence_embeddings = sentence_embeddings.permute(1, 0, 2)  # (m, batch_size, hidden_size)
        sentence_embeddings = self.positional_encoding(sentence_embeddings)  # 加入位置编码
        
        # 使用 mask 传入 Transformer，mask 填充的句子
        transformer_output = self.transformer_encoder(sentence_embeddings, src_key_padding_mask=sentence_mask)  # (m, batch_size, hidden_size)
        transformer_output = transformer_output.permute(1, 0, 2)  # (batch_size, m, hidden_size)
        
        # 交叉注意力机制获取难度多视图表示
        R = self.cross_attention(transformer_output, sentence_mask)  # (batch_size, num_labels, hidden_size)
        
        # 对多视图表示进行 MeanPooling
        pooled_output = self.mean_pooling(R.permute(0, 2, 1)).squeeze(2)  # (batch_size, hidden_size)
        
        # 分类层预测文本的可读性难度
        logits = self.classifier(pooled_output)
        
        logits_pairs = text_to_pairs(logits, pairs)
        
        logits = self.regressor(logits_pairs).squeeze(-1)
        
        return logits


## 定义一个对数据集训练一轮的函数
def train_epoch(model, iterator, optimizer, scheduler, criterion):
    epoch_loss = 0;epoch_acc = 0
    train_corrects = 0;train_num = 0
    model.train()
    for batch_idx, batch_item in enumerate(iterator):
        input_ids, attention_masks, labels, indexs = batch_item
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        # labels = labels.to(device)
        
        pairs = get_pairs(torch.arange(len(indexs)))
        y = text_to_pairs(labels, pairs)
        y_result = y[:,0] - y[:, 1]
        y_result = y_result.reshape(-1).float()
        y_result = y_result.to(device)
        
        optimizer.zero_grad()
        output = model(input_ids, attention_masks, pairs)
        loss = criterion(output, y_result)
        loss.backward()
        
        train_num += len(labels) ## 样本数量
        epoch_loss += loss.item()
        
        # TODO: apply gradient accumulation
        optimizer.step()
        scheduler.step()

    ## 所有样本的平均损失和精度
    epoch_loss = epoch_loss / train_num
    return epoch_loss


## 定义一个对数据集验证一轮的函数
def evaluate(model, dataset, iterator):
    total_num = 0; adjacent_corrects = 0
    min_label, max_label = 0, 2
    
    # Determine the ranking of each index
    rank_dict = {index : [] for index in dataset['index']}
    
    model.eval()
    with torch.no_grad(): # 禁止梯度计算
        true_label = torch.tensor([]); pre_label = torch.tensor([])
        for batch_idx, batch_item in enumerate(iterator):
            input_ids, attention_masks, labels, indexs = batch_item
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            # labels = labels.to(device)
            
            pairs = get_pairs(torch.arange(len(indexs)))[-(len(indexs)-1):]
            
            output = model(input_ids, attention_masks, pairs).to('cpu')
            # loss = criterion(output, labels)
            
            # 将训练集的真实标签和预测差值相加得到测试样本的预测等级
            list_of_pre = [output[deltas] + labels[deltas] for deltas in range(len(output))]
            list_of_pre = [max(min_label, min(max_label, int(round(i.item())))) for i in list_of_pre]
            
            if indexs[-1].item() in rank_dict.keys():
                rank_dict[indexs[-1].item()].extend(list_of_pre)
                
        # 统计 rank_dict 中各个 index 出现出现最多次的标签，即可读性等级
        predict = []
        for index in dataset['index']:
            counter = Counter(rank_dict[index])
            predict.append(counter.most_common(1)[0][0])
            
        for i in range(len(predict)):
            if predict[i] == dataset['labelcode'].values[i] or predict[i] == dataset['labelcode'].values[i]+1 \
            or predict[i] == dataset['labelcode'].values[i]-1:
                adjacent_corrects += 1
        total_num += len(predict) ## 样本数量
        
        # metrics
        true_label = dataset['labelcode'].values
        pre_label = np.array(predict)
        accuracy = accuracy_score(true_label, pre_label)
        adj_acc = float(adjacent_corrects) / total_num
        weighted_f1 = f1_score(true_label, pre_label, average='weighted')
        precision = precision_score(true_label, pre_label, average='weighted')
        recall = recall_score(true_label, pre_label, average='weighted')
        qwk = cohen_kappa_score(true_label, pre_label, weights='quadratic')
            
    return accuracy, adj_acc, weighted_f1, precision, recall, qwk


def split_sentences(text_data, separator):
    """将文本行拆分为句子"""
    text_list = []
    for text in text_data:
        sent_list = text.split(separator)
        text_list.append(sent_list)
    return text_list


def sent_truncate_pad(text, num_steps, sent_max_len, padding_token):
    """截断或填充文本句子"""
    valid_len = len(text)
    if valid_len > num_steps:
        return text[:num_steps]
    return torch.cat([text, torch.tensor([[padding_token] * sent_max_len] * (num_steps - valid_len))], 0)


# 定义自定义 Dataset
class ReadabilityDataset(Dataset):
    def __init__(self, corpus, labels, indexs, tokenizer, max_len=64, max_sentences=None, padding_token=0):
        self.corpus = corpus
        self.labels = labels
        self.indexs = indexs
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_sentences = max_sentences if max_sentences else max([len(t) for t in corpus])
        self.padding_token = padding_token

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        sentences = self.corpus[idx]
        label = self.labels[idx]
        index = self.indexs[idx]
        
        # 将每篇文本中的每个句子转化为 BERT 输入格式
        encode = self.tokenizer(
                    sentences,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors='pt'
                )
        input_ids = encode['input_ids']
        attention_mask = encode['attention_mask']
        # 填充句子，使得每篇文本的句子数量相同
        input_ids = sent_truncate_pad(input_ids, self.max_sentences, self.max_len, self.padding_token)
        attention_mask = sent_truncate_pad(attention_mask, self.max_sentences, self.max_len, self.padding_token)
            
        return input_ids.long(), attention_mask.long(), torch.tensor(label, dtype=torch.long), torch.tensor(index, dtype=torch.long)


path = r"data collect\data_split.csv"
data_df = pd.read_csv(path)
labelMap = {'Ele_Txt' : 0, 'Int_Txt' : 1, 'Adv_Txt' : 2}
data_df['labelcode'] = data_df['label'].map(labelMap)
# 将数据集划分为 训练集， 验证机 和 测试集
train_df, test_df =  train_test_split(data_df, test_size=0.2, random_state=2023)
# 为数据集添加 index ， 方便后续测试集统计预测各个 label 的数量
train_df['index'] = range(len(train_df))
test_df['index'] = range(len(train_df), len(train_df)+len(test_df))
print(len(train_df))
print(len(test_df))


def split_test_subsets(contrast_data, test_df):
    test_subsets_dict = {}
    for i in range(len(test_df)):
        df_new = contrast_data.copy()
        df_new = df_new.append(test_df.iloc[i], ignore_index=True)
        test_subsets_dict[str(i)] = df_new
    return test_subsets_dict


def concat_subsets(subsets_dict, label_col_name, index=None):
    temp_df = pd.DataFrame()
    for key in subsets_dict.keys():
        temp_df = pd.concat([temp_df, subsets_dict[key]], ignore_index=True)
        
    temp_df[label_col_name] = temp_df[label_col_name].astype(int)
    
    if index is not None:
        temp_df[index] = temp_df[index].astype(int)
        
    return temp_df


# 构造用于测试集的对比
_, contrast_data = train_test_split(train_df, test_size=21, stratify=train_df.labelcode, random_state=2023)
test_subsets_dict = split_test_subsets(contrast_data, test_df)
test_data = concat_subsets(test_subsets_dict, 'labelcode', 'index')


# model_name = 'bert'
pretrained_model_name = r"bert-base-uncased"
tokenizer_class = BertTokenizer
# load tokenizer
# model will be loaded for each kth-fold
tokenizer = tokenizer_class.from_pretrained(pretrained_model_name)


# 训练集
train_dataset = []
for text in split_sentences(train_df['split'], ' <_sentence_separator_> '):
    train_dataset.append([line for line in text])
train_label = train_df['labelcode'].tolist()
train_index = train_df['index'].tolist()
# 测试集
test_dataset = []
for text in split_sentences(test_data['split'], ' <_sentence_separator_> '):
    test_dataset.append([line for line in text])
test_label = test_data['labelcode'].tolist()
test_index = test_data['index'].tolist()


# 加载训练好的 BModel
bmodel = BERTDifficultyClassifier(pretrained_model_name, num_labels=3)
bmodel.load_state_dict(torch.load('save model/BModel.pth'))
bmodel.eval()



# load logger
logger = get_logger()
# 超参数
batch_size = 16
epochs = 20
learning_rate = 3e-5
weight_decay = 0.02
warmup_ratio = 0.1
num_labels = 3
time_list = ['First time', 'Second time', 'Third time']
logger.info('********** DSDR Start Run **********')
for i in range(3):
    # set seed for reproducibility
    set_seed(2023+i)
    logger.info(f'********** {time_list[i]} **********')
    # 创建 Dataset 和 DataLoader
    train_dataset = ReadabilityDataset(train_dataset, train_label, train_index, tokenizer, max_len=64, max_sentences=60, padding_token=tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = ReadabilityDataset(test_dataset, test_label, test_index, tokenizer, max_len=64, max_sentences=60, padding_token=tokenizer.pad_token_id)
    test_loader = DataLoader(test_dataset, batch_size=21+1, shuffle=False)
    
    model = TextDifficultyModel(bmodel, num_labels)
    model.to(device)
    
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)  # 预热步数
    
    # AdamW 优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 预热学习率调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.HuberLoss(delta=0.5, reduction='mean')
    
    ## 使用训练集训练模型，验证集测试模型
    for epoch in range(epochs):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion)
        accuracy, adj_acc, weighted_f1, precision, recall, qwk = evaluate(model, test_df, test_loader)
        end_time = time.time()
        logger.info('TRAIN SET    | Epoch: {:.3f} | Loss: {:.5f}'.format(
            epoch+1,
            train_loss
        ))
        logger.info('TESR SET | Epoch: {:.3f} | Accuracy: {:.4f} | Adj_acc: {:.4f} | F1: {:.4f} | P: {:.4f} | R: {:.4f} | QWK: {:.4f}'.format(
            epoch+1,
            accuracy,
            adj_acc,
            weighted_f1,
            precision, recall, qwk
        ))
        print("Epoch:" ,epoch+1 ,"|" ,"Epoch Time: ",end_time - start_time, "s")
        print("Train Loss:", train_loss)
        print("Test. Acc: ",accuracy, "Test. Adj_Acc: ",adj_acc, "Test. F1: ", weighted_f1, 
             "Test. P: ",precision, "Test. R: ",recall, "Test. qwk: ",qwk)