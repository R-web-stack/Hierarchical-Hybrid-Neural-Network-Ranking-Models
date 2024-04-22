#!/usr/bin/env python
# coding: utf-8


import warnings

import re
import os
import pandas as pd
import tqdm
from jiayan import load_lm
from jiayan import CharHMMTokenizer


# ### Text cleaning


def load_text_data(path):
    text_data = []
    grade_list = os.listdir(path)
    for dset in grade_list:
        path_dset = os.path.join(path, dset)
        fname_list = os.listdir(path_dset)
        for fname in fname_list:
            filename = os.path.join(path_dset, fname)
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
                content = re.sub('[(\n)(\r\n)(\t)]', '', content)
                text_data.append(dset + '\t' + content)
    return text_data


def sum_text(path, save):
    data_save = os.path.join(save, 'text_sum.txt')
    text_sum = load_text_data(path)
    
    with open(data_save,"w", encoding='utf-8') as f:
        f.write('\n'.join(text_sum))


path = r'data\CMCC'
save = r'data_collect\CMCC'
sum_text(path, save)


data_df = pd.read_csv(os.path.join(save, 'text_sum.txt'), sep='\t', header=None, names = ['label', 'text'], encoding='utf-8')


## Preprocess Chinese text data, remove unnecessary characters, and perform other operations
def chinese_pre_content(text_data):
    ## Remove letters, remove numbers
    text_data = re.sub('[a-zA-Z]', '', text_data)
    text_data = re.sub("\d+", "", text_data)
    # Replace the English logo with Chinese
    text_data = re.sub(r",", "，", text_data)
    text_data = re.sub(r"\.", "。", text_data)
    text_data = re.sub(r"\?]", "？", text_data)
    text_data = re.sub(r"!", "！", text_data)
    return text_data

data_df['clean_text'] = data_df.text.apply(chinese_pre_content)


# Stroke function
def get_stroke(c):
    # If it returns 0, then the kTotalStrokes field does not exist in Unicode
    strokes = []
    with open(r'data\strokes.txt', 'r') as fr:
        for line in fr:
            strokes.append(int(line.strip()))

    unicode_ = ord(c)

    if 13312 <= unicode_ <= 64045:
        return strokes[unicode_-13312]
    elif 131072 <= unicode_ <= 194998:
        return strokes[unicode_-80338]
    else:
        print("%s should be a CJK char, or not have stroke in unihan data." % c)
        return 'not have stroke'


# Building a stroke number table
def stroke_table(text_data):
    stroke_dict = {}
    count = 0
    for text in text_data.clean_text:
        text = set(text)
        for char in text:
            if char not in stroke_dict:
                stroke = get_stroke(char)
                if isinstance(stroke, int):
                    stroke_dict[char] = get_stroke(char)
                else:
                    print(count)
        count += 1
                
    return stroke_dict


stroke_dict = stroke_table(data_df)


table = pd.DataFrame({'汉字':stroke_dict.keys(), '笔画':stroke_dict.values()})
table.to_excel(
    r'data_collect\CMCC\char_stroke.xlsx',
    index=False,
    header=False)


char_stroke = pd.read_excel(r'data_collect\CMCC\char_stroke.xlsx', header=None)


stroke_dict = {}
for i in range(len(char_stroke)):
    stroke_dict[char_stroke.iloc[i][0]] = char_stroke.iloc[i][1]


# Get all punctuation marks
def punctuation_mark(text_data):
    mark_list = []
    for text in text_data:
        for i in text:
            if i not in stroke_dict and i not in mark_list:
                mark_list.append(i)
    return mark_list


mark_list = punctuation_mark(data_df.clean_text)
print(mark_list)


reserve_mask = ['，', '。', '：', '“', '！', '”', '、', '？', '—', '…', '?', '"', ';', '-', '；', ':', '）', '（', '『', '』', 
               '《', '》', '‘', '’', "'", '．', '」', '【', '】', '－', '∶', '―', '〈', '〉']
abnormal_mask = [i for i in mark_list if i not in reserve_mask]



def clean_abnormal_mask(text_data):
    for text in text_data:
        text = ''.join([i.strip() for i in text if i not in abnormal_mask])
        return text_data


data_df['clean_text'] = data_df.clean_text.apply(clean_abnormal_mask)


# ### Divide the text into sentences and words


def chineseCutSentence(doc):
    # sentences = re.split('。*|！|\!|\.|？|\?',doc)
    #sentences = re.split('[\.\!\?]+|[！。？]+', doc)
    sentences = re.split('[。|？|！|?|!|…|．|.]', doc)
    a=''
    while a in sentences:
        sentences.remove(a)
    return sentences


lm = load_lm(r'D:\anaconda\Lib\site-packages\jiayan\jiayan_models\jiayan.klm')
tokenizer = CharHMMTokenizer(lm)
def doc_split(text_data):
    text = chineseCutSentence(text_data)
    doc = []
    for sent in text:
        if len(sent) > 1:
            sent_cut = list(tokenizer.tokenize(sent))
            word_list = [i.strip() for i in sent_cut]
            word_list.append("<_sentence_separator_>")
            doc += word_list
            
    if len(doc) > 1:
        #remove last sentence separator
        doc = doc[:-1]
    
    return ' '.join(doc)


data_df['split'] = data_df.clean_text.apply(doc_split)


data_df.to_csv(r'data_collect\CMCC\data_clean.csv', index=False)

# ### Clean up punctuation and stop words in text


## Remove some punctuation marks from the segmented text and try to preserve the sentence structure as much as possible
def del_pun(text_data):
    pun_mask = ['“', '”', '—', '…', '"', '-', '）', '（', '『', '』', 
               '《', '》', '‘', '’', "'", '」', '【', '】', '－', '―', '〈', '〉']
    text_list = text_data.split(' ')
    text_list = [word.strip() for word in text_list if word not in pun_mask]
    ## The processed words are concatenated into strings using spaces
    text = " ".join(text_list)
    return text


data_df['del_pun'] = data_df.split.apply(del_pun)


data_df.to_csv(r'data_collect\CMCC\data_split.csv', index=False)
