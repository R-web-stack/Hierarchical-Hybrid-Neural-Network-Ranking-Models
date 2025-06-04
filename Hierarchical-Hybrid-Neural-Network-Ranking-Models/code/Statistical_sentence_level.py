#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import ast
from collections import Counter
from d2l import torch as d2l

data_df = pd.read_csv("sentence level\predicted_sentence_levels_29.csv")

path = r"data collect\data_split.csv"
original_df = pd.read_csv(path)
labelMap = {'Ele_Txt' : 0, 'Int_Txt' : 1, 'Adv_Txt' : 2}
original_df['labelcode'] = original_df['label'].map(labelMap)
original_df['index'] = range(len(original_df))

merge_df = pd.merge(original_df, data_df, on='index', how='inner')


def create_sentences_corpus(df, text_col_name):
    sentences_level_dict = {'index': [], 'sentence': [], 'label': []}
    for row in range(len(df)):
        sentences_list = df.iloc[row][text_col_name].split(' <_sentence_separator_> ')
        for index, level in enumerate(ast.literal_eval(df.iloc[row]['sentences_level'])):
            sentences_level_dict['index'].append(df.iloc[row]['index'])
            sentences_level_dict['sentence'].append(sentences_list[index])
            sentences_level_dict['label'].append(level)
    return pd.DataFrame(sentences_level_dict)


sentence_df = create_sentences_corpus(merge_df, 'split')


sentence_df.to_csv(r'final sentence level/sentence corpus.csv', index=False)


sentence_corpus = pd.read_csv(r'final sentence level/sentence corpus.csv')


# 假设df为你的数据集，包含'sentence'和'label'两列
# 首先计算每个句子的长度
sentence_corpus['sentence_length'] = sentence_corpus['sentence'].apply(len)

# 然后根据label分组，计算每个label的句子平均长度
average_lengths = sentence_corpus.groupby('label')['sentence_length'].mean()

# 打印结果
print(average_lengths)