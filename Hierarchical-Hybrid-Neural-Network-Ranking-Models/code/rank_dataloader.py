# coding=utf-8

import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import random
import itertools
from tqdm import tqdm
import torch
from datasetloader import split_sentences, tokenize, Vocab, mytorchtext, load_array


# The iterator for constructing a dataset is actually an index iterator
def array_dataloader(dataset, shuffle):
    data_index = list(range(len(dataset)))
    if shuffle:
        r=random.random
        random.seed(2023)
        random.shuffle(data_index, random=r)
    index_iterator = itertools.product(data_index)
    return index_iterator


def split_subsets(dataset, label_col, subset_num, shuffle):
    # Number of statistical categories
    categories_num = len(dataset[label_col].value_counts())
    
    # Construct datasets for each category
    categories_set_dict = {}
    category_prefix = 'category_'   # Category dataset key prefix
    for i in range(categories_num):
        categories_set_dict[category_prefix + str(i)] = dataset[dataset[label_col] == i]
        
    # Construct a class iterator
    categories_set_loaders = {}
    for name in categories_set_dict.keys():
        categories_set_loaders[name] = array_dataloader(categories_set_dict[name], shuffle)
    
    # Construct multiple data subsets, each containing only one sample from each category
    subsets_dict = {}
    for j in tqdm(range(subset_num)):
        temp_df = pd.DataFrame()
        for name in categories_set_dict.keys():
            try:
                index = next(categories_set_loaders[name])
            except StopIteration:
                categories_set_loaders[name] = array_dataloader(categories_set_dict[name], shuffle)
                index = next(categories_set_loaders[name])
                data_row = categories_set_dict[name].iloc[index]
                temp_df = pd.concat([temp_df, data_row.to_frame().T], ignore_index=True)
            else:
                data_row = categories_set_dict[name].iloc[index]
                temp_df = pd.concat([temp_df, data_row.to_frame().T], ignore_index=True)
        subsets_dict[str(j)] = temp_df
    
    return subsets_dict
    
    
def concat_subsets(subsets_dict, label_col_name, index=None):
    temp_df = pd.DataFrame()
    for key in tqdm(subsets_dict.keys()):
        temp_df = pd.concat([temp_df, subsets_dict[key]], ignore_index=True)
        
    temp_df[label_col_name] = temp_df[label_col_name].astype(int)
    
    if index is not None:
        temp_df[index] = temp_df[index].astype(int)
        
    return temp_df
    

def rankmodel_load_data(train_data, test_data, text_col_name, label_col_name, index, batch_size, 
              sent_num_steps, token_num_steps, vocab=None, min_freq=None):
    train_texts_split = []
    for text in split_sentences(train_data[text_col_name], ' <_sentence_separator_> '):
        train_texts_split.append(tokenize(text, token='word'))
        
    test_texts_split = []
    for text in split_sentences(test_data[text_col_name], ' <_sentence_separator_> '):
        test_texts_split.append(tokenize(text, token='word'))
    
    if vocab is None:
        vocab = Vocab(train_texts_split, min_freq, reserved_tokens=['<pad>'])
    else:
        vocab = vocab
    
    train_texts_features, train_textsSent_valid_len, train_textsToken_valid_len = mytorchtext(train_texts_split, vocab, sent_num_steps, token_num_steps)
    
    test_texts_features, test_textsSent_valid_len, test_textsToken_valid_len = mytorchtext(test_texts_split, vocab, sent_num_steps, token_num_steps)
        
    train_iter = load_array((torch.tensor(train_texts_features), 
                             torch.tensor(train_textsSent_valid_len), 
                             torch.tensor(train_textsToken_valid_len),
                             torch.tensor(train_data[label_col_name].to_numpy()), 
                             torch.tensor(train_data[index].to_numpy())), batch_size, is_train=False)
    
    test_iter = load_array((torch.tensor(test_texts_features), 
                            torch.tensor(test_textsSent_valid_len), 
                            torch.tensor(test_textsToken_valid_len),
                            torch.tensor(test_data[label_col_name].to_numpy()), 
                            torch.tensor(test_data[index].to_numpy())), batch_size, is_train=False)
        
    return train_iter, test_iter, vocab