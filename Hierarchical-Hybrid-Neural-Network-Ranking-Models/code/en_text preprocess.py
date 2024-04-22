#!/usr/bin/env python
# coding: utf-8


import warnings
warnings.filterwarnings('ignore')

import re
import os
import pandas as pd
import spacy
import chardet


def loadDocsFromFolder(folder, category):
    text_data = []
    fname_list = os.listdir(folder)
    for fname in fname_list:
        if fname.endswith(".txt"):
            filename = os.path.join(folder, fname)
            # Detect encoding when reading files
            with open(filename, 'rb') as file:
                rawdata = file.read()
                result = chardet.detect(rawdata)
                encoding = result['encoding']
            
            with open(filename, 'r', encoding=encoding) as f:
                content = f.read()
                content = re.sub('[(\n)(\r\n)(\t)]', '', content)
            text_data.append(category + '\t'  + content)
    return text_data


def en_sum_text(path, save, data_type):
    data_path = path[data_type]
    data_save = os.path.join(save, data_type + '_sum.txt')
    text_sum = loadDocsFromFolder(data_path, data_type)
    
    with open(data_save,"w", encoding='gb18030') as f:
        f.write('\n'.join(text_sum))


data_path = {'Ele_Txt' : r"data\OneStopEnglishCorpus-master\Texts-SeparatedByReadingLevel\Ele-Txt",
'Int_Txt' : r"data\OneStopEnglishCorpus-master\Texts-SeparatedByReadingLevel\Int-Txt\Int-Txt",
'Adv_Txt' : r"data\OneStopEnglishCorpus-master\Texts-SeparatedByReadingLevel\Adv-Txt"}
save = r"data_collect\OSP"
en_sum_text(data_path, save, 'Ele_Txt')
en_sum_text(data_path, save, 'Int_Txt')
en_sum_text(data_path, save, 'Adv_Txt')


data_Ele = pd.read_csv(os.path.join(save, 'Ele_Txt_sum.txt'), sep='\t', header=None, names = ['label', 'text'], encoding='gb18030')
data_Int = pd.read_csv(os.path.join(save, 'Int_Txt_sum.txt'), sep='\t', header=None, names = ['label', 'text'], encoding='gb18030')
data_Adv = pd.read_csv(os.path.join(save, 'Adv_Txt_sum.txt'), sep='\t', header=None, names = ['label', 'text'], encoding='gb18030')
data_df = pd.concat([data_Ele, data_Int, data_Adv])


class Tokenizer:
    def __init__(self,lang="en_core_web_sm"):
        self.spacy_nlp = spacy.load('en_core_web_sm')

    def tokenize(self, inputs):
        return [x.text for x in self.spacy_nlp.tokenizer(inputs) if x.text != " "]

    def split_sentences(self, inputs):
        return [x.text for x in self.spacy_nlp(inputs).sents if x.text != " "]

tokenizer = Tokenizer()
def doc_split(text_data):
    text = tokenizer.tokenize(text_data)
    doc = []
    sent = []
    for token in text:
        if token not in ['"', "'", '-', '–', '‘', '’', '“', '”', '…', ' ', '']:
            if token in ['!', '.', '?', '？', '！']:
                if len(sent) > 1:
                    sent.append("<_sentence_separator_>")
                    doc += sent
                    sent = []
                else:
                    sent = []
            else:
                sent.append(token.strip().lower())
                
                
    if len(doc) > 1:
        #remove last sentence separator
        doc = doc[:-1]
    
    return ' '.join(doc)


data_df['split'] = data_df.text.apply(doc_split)


data_df.to_csv(r'data_collect\OSP\data_split.csv', index=False)
