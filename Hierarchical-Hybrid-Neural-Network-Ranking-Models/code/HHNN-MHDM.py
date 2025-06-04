#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(r"python code")

import os
import time
import math
import pandas as pd
from dataLoader import load_data
import torch
import torch.optim as optim
from model_layer import *
from utils import get_logger, set_seed
from train import evaluate, train_epoch, Trainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
from d2l import torch as d2l

# 模型加载选择GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))


path = r"data collect\data_split.csv"
data_df = pd.read_csv(path)
labelMap = {'Ele_Txt' : 0, 'Int_Txt' : 1, 'Adv_Txt' : 2}
data_df['labelcode'] = data_df['label'].map(labelMap)


# arguments
parser = argparse.ArgumentParser()

# required
parser.add_argument('--corpus_name',
                    default='OSP',
                    type=str,
                    help='name of the corpus to be trained on')

parser.add_argument('--data_dir',
                    default="data collect\data_split.csv",
                    type=str,
                    help='path to a data directory')

parser.add_argument('--num_class',
                    default=3,
                    type=int,
                    help="number of readability levels for the dataset")

parser.add_argument('--save_sentence_dir',
                    default="sentence level\predicted_sentence_levels",
                    type=str,
                    help='path of predicted sentence levels')

# model parameter
parser.add_argument('--is_GRU',
                    default=True,
                    type=bool,
                    help="GRU or LSTM")

parser.add_argument('--d_model',
                    default=400,
                    type=int,
                    help="model dimension")

parser.add_argument('--n_h',
                    default=8,
                    type=int,
                    help="the number of attention heads")

parser.add_argument('--hidden_dim',
                    default=200,
                    type=int,
                    help="RNN output dimension")

parser.add_argument('--layer_dim',
                    default=1,
                    type=int,
                    help="RNN layer dimension")

parser.add_argument('--bidirectional',
                    default=True,
                    type=bool,
                    help="whether or not to adopt bidirectional RNN")

parser.add_argument('--n_filters',
                    default=200,
                    type=int,
                    help="context dimension -- CNN channel")

parser.add_argument('--n_h_cnn',
                    default=10,
                    type=int,
                    help="number of contextual attention heads")

parser.add_argument('--dropout',
                    default=0.3,
                    type=float,
                    help="dropout rate")

parser.add_argument('--n_encoderLayer',
                    default=1,
                    type=int,
                    help="number of encoderLayer")

# optional
parser.add_argument('--seed',
                    default=2023,
                    type=int,
                    help="seed value")
parser.add_argument('--epochs',
                    default=30,
                    type=int,
                    help="number of epochs to train")
parser.add_argument('--model',
                    default='HHNN_Pseudo_Label_Pre_Sent',
                    type=str,
                    help="model to use for classification or regression")
parser.add_argument('--batch_size',
                    default=8,
                    type=int,
                    help="size of a batch for each iteration both on training and evaluation process")
parser.add_argument('--num_sents',
                    default=60,
                    type=int,
                    help="sentence count")
parser.add_argument('--num_tokens',
                    default=50,
                    type=int,
                    help="token count")
parser.add_argument('--min_freq',
                    default=0,
                    type=int,
                    help="minimum frequency of tokens in the vocabulary list")
parser.add_argument('--learning_rate',
                    default=1e-3,
                    type=float,
                    help="learning rate to train")
parser.add_argument('--weight_decay',
                    default=5e-4,  #  1e-4
                    type=float,
                    help="weight_decay rate to train")
parser.add_argument('--tsa',
                    default='linear_schedule',  # 'linear_schedule'  'exp_schedule'  'log_schedule'
                    type=str,
                    help="Training Signal Annealing")
parser.add_argument('--uda_softmax_temp',
                    default=0.85,
                    type=float,
                    help="uda_softmax_temp rate to train")
parser.add_argument('--uda_confidence_thresh',
                    default=0.45,
                    type=float,
                    help="uda_confidence_thresh rate to train")
parser.add_argument('--device',
                    default='cuda',
                    type=str,
                    help="set to 'cuda' to use GPU. set to 'cpu' otherwise")
parser.add_argument('--n_eval_per_epoch',
                    default=1,
                    type=int,
                    help=("number of evaluation and save for each epoch.",
                          "allows understanding distribution of discrepency between train and validation set"))
parser.add_argument('--one_fold',
                    default=False,
                    type=bool,
                    help="whether or not to train only on the first fold out of k folds")
parser.add_argument('--do_evaluate',
                    default=True,
                    type=bool,
                    help="whether or not to evaluate the training, only train.csv needed to process")

args = parser.parse_args(args=[])
print(args)


class ClassifyModel(customizedModule):
    def __init__(self, args, Encoder, vocabe):

        super(ClassifyModel, self).__init__()
        self.imput_dim = args.d_model
        self.embeddings = nn.Embedding(len(vocab.idx_to_token), self.imput_dim)
        self.pad_idx = vocab.token_to_idx['<pad>']
        self.rnn = Bi_Rnn_Layer(args.is_GRU, self.imput_dim, args.hidden_dim, args.layer_dim, 
                                bidirectional=args.bidirectional, batch_first=True)
        self.cnn_context = Mdim_CNN_Context(self.imput_dim, args.n_filters, 3, 1, args.n_h_cnn)
        self.Encoder = Encoder
        self.init_mBloSA()
        
        # 多头难度矩阵
        self.difficulty_matrix = MHDifficultyMatrix(args, self.imput_dim, args.n_h)
        
        # 多视图表示融合
        self.s2tSA = s2tSA(self.imput_dim)
        
        self.classifier = nn.Linear(self.imput_dim, args.num_class)

    def init_mBloSA(self):
        self.f_W1 = self.customizedLinear(self.imput_dim * 2, self.imput_dim, activation=nn.ReLU())
        self.f_W2 = self.customizedLinear(self.imput_dim * 2, self.imput_dim)

    def forward(self, words, nSents, nWords, sentence_features=None):
        embeds = self.embeddings(words)
        
        hAllWords = self.rnn(embeds, nWords)
        cnn_hAllWords = self.cnn_context(hAllWords, nWords)

        attention_output = self.Encoder(cnn_hAllWords, mask=nSents)
        # (batch, n, word_dim * 3) -> (batch, n, word_dim)
        fusion = self.f_W1(torch.cat([cnn_hAllWords, attention_output], dim=2))
        G = F.sigmoid(self.f_W2(torch.cat([cnn_hAllWords, attention_output], dim=2)))
            
        # (batch, n, word_dim)
        u = G * fusion + (1 - G) * cnn_hAllWords
        
        # 特征融合
        if sentence_features is not None:
            u, features_attention_weights = self.FFA(u, sentence_features, nSents)

        output, sentence_weights = self.s2tSA(u, nSents)
        
        logits = self.classifier(output)
        return logits, sentence_weights, u


def init_weights(m):
    if type(m) in (nn.Linear, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        
    if type(m) in (nn.LSTM, nn.GRU):
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])


# load logger
logger = get_logger()
time_list = ['First time', 'Second time', 'Third time']
logger.info(f'********** {args.model} Start Run **********')
for k_cnt in range(3):
    # set seed for reproducibility
    set_seed(args.seed + k_cnt)
    logger.info(f'********** {time_list[k_cnt]} **********')
    
    # 为数据集添加 index ， 方便后续测试集统计预测各个 label 的数量
    data_df['index'] = range(len(data_df))
    
    train_iter, _, vocab = load_data(data_df, 'split', 'labelcode',  'index', 
                                 args.batch_size, args.num_sents, args.num_tokens, min_freq=args.min_freq)
    
    # Prepare model
    multiHeadedAttention = MultiHeadAttention(key_size=args.d_model, query_size=args.d_model, value_size=args.d_model, 
                                              num_hiddens=args.d_model, num_heads=args.n_h, dropout=args.dropout)
    positionwiseFeedForward = PositionwiseFeedForward(args.d_model, args.d_model * 2)
    encoderLayer = EncoderLayer(args.d_model, multiHeadedAttention, positionwiseFeedForward, args.dropout)
    encoder = Encoder(encoderLayer, args.n_encoderLayer)
    
    model = ClassifyModel(args, encoder, vocab)
    model.apply(init_weights)
        
    model = model.to(args.device)
    
    ## Adam优化,交叉熵作为损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(reduction='none')
    sent_criterion = nn.KLDivLoss(reduction='none')
    
    data_iter = {'train': train_iter}
    
    trainer = Trainer(args, model, optimizer, data_iter, criterion, sent_criterion, k_cnt)
    trainer.train(logger)