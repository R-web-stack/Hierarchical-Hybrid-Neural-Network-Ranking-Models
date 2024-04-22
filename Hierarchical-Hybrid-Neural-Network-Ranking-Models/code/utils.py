# coding=utf-8

import warnings
warnings.filterwarnings('ignore')

import logging
import time
import os
import random
from itertools import permutations

import torch
import numpy as np


# Get all permutation pairs out of an array
def get_pairs(arr):
    return torch.tensor(list(permutations(arr, 2)))

def text_to_pairs(data, pairs):
    pairsFlat = pairs.reshape(-1)
    data_pairs = data[pairsFlat].split(pairs.size(0), dim=0)
    data_pairs = torch.stack(data_pairs, dim=0).reshape(pairs.size(0), -1)
    return data_pairs

def save(args, model, info, k_cnt, l_cnt):
    # save model
    os.makedirs(args.save_dir, exist_ok=True)
    save_dir = os.path.join(args.save_dir, f'{args.corpus_name}.{args.model}.{k_cnt}.{l_cnt}')
    os.makedirs(save_dir, exist_ok=True)
    # save model
    torch.save(model.state_dict(), os.path.join(save_dir, f'pytorch_model_mf{args.min_freq}.pt'))
#     # DEPRECATED: we decided to have global k-fold to keep consistency on different models
#     # save dataframes used for each train / valid / test set
#     train_df.to_csv(os.path.join(save_dir, 'train_df.csv'), index=False)
#     test_df.to_csv(os.path.join(save_dir, 'test_df.csv'), index=False)
    with open(os.path.join(save_dir, 'info.txt'), 'w') as f:
        f.write(' '.join(map(str, info)))

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

# set seed values
def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    