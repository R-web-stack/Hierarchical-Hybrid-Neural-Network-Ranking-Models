#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')

import os
import time
import math 
import pandas as pd
from rank_dataloader import *
from datasetloader import load_data
import torch
import torch.optim as optim
from model_layer import *
from utils import get_logger, get_pairs, text_to_pairs, save, set_seed
from train import rank_evaluate, rank_train_epoch, Rank_Trainer
from sklearn.model_selection import train_test_split
import argparse



# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))


# arguments
parser = argparse.ArgumentParser()

# required
parser.add_argument('--corpus_name',
                    default='OSP',
                    type=str,
                    help='name of the corpus to be trained on')

parser.add_argument('--data_dir',
                    default='data_collect',
                    type=str,
                    help='path to a data directory')

parser.add_argument('--num_class',
                    default=3,
                    type=int,
                    help="number of readability levels for the dataset")

parser.add_argument('--num_rank_class',
                    default=4,
                    type=int,
                    help="the number of readability level sorting for the dataset")

parser.add_argument('--num_subset',
                    default=78,
                    type=int,
                    help="number of test subset")

parser.add_argument('--save_dir',
                    default='save model\checkpoint_HHNNRM',
                    type=str,
                    help='path to save model')

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
                    default=20,
                    type=int,
                    help="number of epochs to train")
parser.add_argument('--model',
                    default='HHNNRM',
                    type=str,
                    help="model to use for classification or regression")
parser.add_argument('--batch_size',
                    default=3,
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
                    default=5e-4,  # 1e-3
                    type=float,
                    help="weight_decay rate to train")
parser.add_argument('--device',
                    default='cuda',
                    type=str,
                    help="set to 'cuda' to use GPU. set to 'cpu' otherwise")
parser.add_argument('--n_eval_per_epoch',
                    default=3,
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



class RankModel(customizedModule):
    def __init__(self, args, Encoder, vocabe):

        super(RankModel,self).__init__()
        self.imput_dim = args.d_model
        self.embeddings = nn.Embedding(len(vocab.idx_to_token), self.imput_dim)
        self.pad_idx = vocab.token_to_idx['<pad>']
        self.rnn = Bi_Rnn_Layer(args.is_GRU, self.imput_dim, args.hidden_dim, args.layer_dim, 
                                bidirectional=args.bidirectional, batch_first=True)
        self.cnn_context = Mdim_CNN_Context(self.imput_dim, args.n_filters, 3, 1, args.n_h_cnn)
        self.Encoder = Encoder
        self.init_mBloSA()
        self.s2tSA = s2tSA(self.imput_dim)
        self.classifier = nn.Sequential(nn.Linear(self.imput_dim*2, self.imput_dim),
                      nn.ReLU(),
                      nn.Linear(self.imput_dim, args.num_rank_class))

    def init_mBloSA(self):
        self.f_W1 = self.customizedLinear(self.imput_dim * 2, self.imput_dim, activation=nn.ReLU())
        self.f_W2 = self.customizedLinear(self.imput_dim * 2, self.imput_dim)

    def forward(self, words, nSents, nWords, pairs):
        embeds = self.embeddings(words)
        
        hAllWords = self.rnn(embeds, nWords)
        cnn_hAllWords = self.cnn_context(hAllWords, nWords)

        attention_output = self.Encoder(cnn_hAllWords, mask=nSents)
        # (batch, n, word_dim * 3) -> (batch, n, word_dim)
        fusion = self.f_W1(torch.cat([cnn_hAllWords, attention_output], dim=2))
        G = F.sigmoid(self.f_W2(torch.cat([cnn_hAllWords, attention_output], dim=2)))
        # (batch, n, word_dim)
        u = G * fusion + (1 - G) * cnn_hAllWords

        # logits = self.s2t(u, context, nSents)
        logits = self.s2tSA(u)
        
        logits_pairs = text_to_pairs(logits, pairs)
        
        logits = self.classifier(logits_pairs)
        return logits



def init_weights(m):
    if type(m) in (nn.Linear, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        
    if type(m) in (nn.LSTM, nn.GRU):
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])



# form paths to df
train_df_paths = os.path.join(args.data_dir, f'{args.corpus_name}.train.csv')
if args.do_evaluate:
    test_df_paths = os.path.join(args.data_dir, f'{args.corpus_name}.test.csv')


pretrain_path=  "save model\checkpoint_HHNN\OSP.HHNN.0.14\pytorch_model_mf0.pt"



# ### main

# After subtracting the original label, it is -1:2 and 1:2. Due to training needs, conversion is required.
label_map = {-2.0 : 0, -1.0 : 1, 1 : 2, 2.0 : 3}

# load logger
logger = get_logger()
time_list = ['First time', 'Second time', 'Third time']
logger.info(f'********** {args.model} Start Run **********')
for k_cnt in range(3):
    # set seed for reproducibility
    set_seed(args.seed+k_cnt)
    logger.info(f'********** {time_list[k_cnt]} **********')
    train_df = pd.read_csv(train_df_paths)
    test_df = pd.read_csv(test_df_paths)
    
    _, _, _, vocab = load_data(train_df, test_df, 'split', 'labelcode', 
                                         16, args.num_sents, args.num_tokens, min_freq=args.min_freq)
    
    # Add an index to the dataset to facilitate the subsequent statistical prediction of the number of labels in the test set.
    train_df['index'] = range(len(train_df))
    test_df['index'] = range(len(test_df))
    
    class_maximum = train_df['labelcode'].value_counts().max()
    
    train_subsets_dict = split_subsets(train_df, 'labelcode', class_maximum, True)
    test_subsets_dict = split_subsets(test_df, 'labelcode', args.num_subset, True)
    
    train_data = concat_subsets(train_subsets_dict, 'labelcode', 'index')
    test_data = concat_subsets(test_subsets_dict, 'labelcode', 'index')
    
    train_iter, test_iter, vocab = rankmodel_load_data(train_data, test_data, 'split', 'labelcode', 'index', 
                                         args.batch_size, args.num_sents, args.num_tokens, vocab=vocab)
    
    # Prepare model
    multiHeadedAttention = MultiHeadAttention(key_size=args.d_model, query_size=args.d_model, value_size=args.d_model, 
                                              num_hiddens=args.d_model, num_heads=args.n_h, dropout=args.dropout)
    positionwiseFeedForward = PositionwiseFeedForward(args.d_model, args.d_model * 2)
    encoderLayer = EncoderLayer(args.d_model, multiHeadedAttention, positionwiseFeedForward, args.dropout)
    encoder = Encoder(encoderLayer, args.n_encoderLayer)
    
    model = RankModel(args, encoder, vocab)
    model.apply(init_weights)
    
    c_model_state_dict = torch.load(pretrain_path)
    model_state_dict = model.state_dict()
    
    # Update the parameter dictionary of the new model and load the previously trained parameters.
    for name, param in c_model_state_dict.items():
        if name not in model_state_dict:
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        model_state_dict[name].copy_(param)
        
    model = model.to(args.device)
    
    ## Adam optimization, cross entropy as loss function
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    data_set = {'train': train_df, 'test': test_df}
    data_iter = {'train': train_iter, 'test': test_iter}
    
    trainer = Rank_Trainer(args, model, optimizer, data_set, data_iter, label_map, criterion, k_cnt)
    trainer.train(logger)
    
    if args.one_fold:
        break



# Save parameters to a file
with open(os.path.join(args.save_dir, 'parsed_args.txt'), 'w') as file:
    file.write(str(args))
