#!/usr/bin/env python
# coding: utf-8


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import jieba
import re
import os
import thulac

import math
from tqdm import tqdm
import stanza
from hanlp_restful import HanLPClient
from itertools import chain
import hanlp


def load_text_data(path):
    text_data = []
    grade_list = os.listdir(path)
    for dset in grade_list:
        path_dset = os.path.join(path, dset)
        publish_list = os.listdir(path_dset)
        for publish in publish_list:
            if publish != '课外阅读':
                path_publish = os.path.join(path_dset, publish)
                fname_list = os.listdir(path_publish)
                for fname in fname_list:
                    filename = os.path.join(path_publish, fname)
                    with open(filename, 'r', encoding='utf-8') as f:
                        content = f.read()
                        content = re.sub('[(\n)(\r\n)(\t)]', '', content)
                        text_data.append(dset + '\t' + content)
    return text_data


stop_words1 = pd.read_csv(r"data\stopwords-master\hit_stopwords.txt",
                         header=None,names = ["text"], sep='\n',quoting=3)
stop_words2 = pd.read_csv(r"data\中文停用词库.txt",
                         header=None,names = ["text"], sep='\n',quoting=3)
stop_words = pd.concat([stop_words1, stop_words2])


## Preprocess Chinese text data by removing unnecessary characters, segmenting words, removing stop words, and other operations
def chinese_pre(text_data):
    ## Remove stop words and extra spaces
    text_data = [word[0].strip() for word in text_data if word[0] not in stop_words.text.values]
    ## The processed words are concatenated into strings using spaces
    text_data = " ".join(text_data)
    return text_data

def chinese_pre_content(text_data):
    ## Remove letters and numbers
    text_data = re.sub('[a-zA-Z]', '', text_data)
    text_data = re.sub("\d+", "", text_data)
    # Replace the English logo with Chinese
    text_data = re.sub(r",", "，", text_data)
    text_data = re.sub(r"\.", "。", text_data)
    text_data = re.sub(r"\?]", "？", text_data)
    text_data = re.sub(r"!", "！", text_data)
    ## Word segmentation and part of speech tagging
    text_data = thu1.cut(text_data,text=False) 
    return text_data


def chinese_pre_cutword(text_data):
    text = [word[0].strip() for word in text_data if word[1] != 'w']
    text = " ".join(text)
    return text


def chinese_pre_pos(text_data):
    ## Part of speech tagging
    pos = [word[1].strip() for word in text_data if word[1] != 'w']
    pos = " ".join(pos)
    return pos


def train_test(path, save):
    save = os.path.join(save, '_sum_thulac.txt')
    text_sum = load_text_data(path)
    
    with open(save,"w", encoding='utf-8') as f:
        f.write('\n'.join(text_sum))
        
    data_df = pd.read_csv(save, sep='\t', header=None, names = ['label', 'text'], encoding='utf-8')
    
    data_df["cutword_pos"] = data_df.text.apply(chinese_pre_content)
    data_df["cutword"] = data_df.cutword_pos.apply(chinese_pre_cutword)
    data_df["pos"] = data_df.cutword_pos.apply(chinese_pre_pos)
    data_df["cutword_filt"] = data_df.cutword_pos.apply(chinese_pre)
    return data_df


thu1 = thulac.thulac()
path = r'data\CLT'
save = r'data_collect\CLT'
data = train_test(path, save)

data.to_csv(r'data_collect\CLT\CLT_thulac.csv', index=False)


temp_list = []
for text in data['cutword_pos']:
    for word in text:
        if word[1] == 'x':
            temp_list.append(word[0])
            
print(temp_list)



def chinese_pre_cutword_x(text_data):
    text = [word[0].strip() for word in text_data if word[1] != 'w' and word[1] != 'x']
    text = " ".join(text)
    return text



def chinese_pre_pos_x(text_data):
    pos = [word[1].strip() for word in text_data if word[1] != 'w' and word[1] != 'x']
    pos = " ".join(pos)
    return pos


def chinese_pre_x(text_data):
    text_data = [word[0].strip() for word in text_data if word[0] not in stop_words.text.values and word[1] != 'w' and word[1] != 'x']
    text_data = " ".join(text_data)
    return text_data

data["cutword"] = data.cutword_pos.apply(chinese_pre_cutword_x)
data["pos"] = data.cutword_pos.apply(chinese_pre_pos_x)
data["cutword_filt"] = data.cutword_pos.apply(chinese_pre_x)



data.to_csv(r'data_collect\CLT\CLT_thulac.csv', index=False)


data_df = pd.read_csv(r'data_collect\CLT\CLT_thulac.csv')


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
    for text in tqdm(text_data):
        text = list(''.join(text.split(' ')))
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


stroke_dict = stroke_table(data_df.cutword)


table = pd.DataFrame({'汉字':stroke_dict.keys(), '笔画':stroke_dict.values()})
table.to_excel(
    r'data_collect\CLT\char_stroke.xlsx',
    index=False,
    header=False)



char_stroke = pd.read_excel(r'data_collect\CLT\char_stroke.xlsx', header=None)

stroke_dict = {}
for i in range(len(char_stroke)):
    stroke_dict[char_stroke.iloc[i][0]] = char_stroke.iloc[i][1]


# Abnormal characters in statistical text
def clean_text(text_data):
    unusual_char = []
    for text in tqdm(text_data):
        text = set(list(''.join(text.split(' '))))
        for char in text:
            try:
                temp = stroke_dict[char]
            except KeyError:
                unusual_char.append(char)
            else:
                continue
            
    return unusual_char


unusual_char = clean_text(data_df.cutword)


unusual_char_df = pd.DataFrame({'char':unusual_char})



# Cleaning abnormal characters
def clean_text2(text_data, pos_data):
    cutword = []
    pos_list = []
    for i in tqdm(range(len(text_data))):
        text = text_data.iloc[i].split(' ')
        pos = pos_data.iloc[i].split(' ')
        temp_text = []
        temp_pos = []
        for i, j in zip(text, pos):
            if i not in unusual_char:
                temp_text.append(i)
                temp_pos.append(j)
        cutword.append(' '.join(temp_text))
        pos_list.append(' '.join(temp_pos))
            
    return cutword, pos_list 


data_df['cutword'], data_df['pos'] = clean_text2(data_df.cutword, data_df.pos)


# # Investigating Chinese Text Readability: Linguistic Features, Modeling, and Validation
# ## 词汇类指标
# ### 词汇数量

# In[26]:


# 字数
def char_num(text_data):
    text_data = ''.join(text_data.split(' '))
    return len(text_data)

# 词数
def word_num(text_data):
    text_data = text_data.split(' ')
    return len(text_data)


data_df['char_num'] = data_df.cutword.apply(char_num)
data_df['word_num'] = data_df.cutword.apply(word_num)


# ### 词汇丰富度


# 相异词数比率
def TTR(text_data):
    text_data = text_data.split(' ')
    text_data_set = set(text_data)
    return len(text_data_set) / len(text_data)


data_df['TTR'] = data_df.cutword.apply(TTR)


# n/名词  np/人名  ns/地名  ni/机构名  nz/其它专名  
# m/数词  q/量词  mq/数量词  t/时间词  f/方位词  s/处所词  
# v/动词  vm/能愿动词  vd/趋向动词  a/形容词  d/副词  
# h/前接成分  k/后接成分  i/习语  j/简称  
# r/代词  c/连词  p/介词  u/助词  y/语气助词  
# e/叹词  o/拟声词  g/语素  w/标点  x/其它


# 实词密度  
def real_word(text_data):
    pos = text_data.split(' ')
    real_words = [
        tag for tag in pos if tag in [
            'n', 'np', 'ns', 'ni', 'nz', 'm', 'q', 'mq', 't', 'f', 's', 'v',
            'vm', 'vd', 'a', 'j'
        ]
    ]
    return len(real_words) / len(pos)


data_df['real_word'] = data_df.pos.apply(real_word)

# ### 词汇频率


# 实词频对数平均(Real Word Frequency Logarithmic Average) 
def RWFLA(text_data, pos_data):
    dict_sum = {}  # 各个实词在语料库的总数表
    dict_sum_frelog = {}   # 各个实词在语料库占所有词的频率取对数表
    word_sum = 0   # 语料库词总数
    text_pos = []  # 文本词性标注配对表
    RWFLA_list = []   # 各个文本中实词频对数平均表
    
    # 对所有文本进行词性标注
    for i in range(len(text_data)):
        text = text_data.iloc[i].split(' ')
        pos = pos_data.iloc[i].split(' ')
        text_pos.append(dict(zip(text, pos)))
        # 统计各个文本中较有意义的实词
        word_sum += len(text)
        for word, tag in zip(text, pos):
            if tag in ['n', 'np', 'ns', 'ni', 'nz', 'm', 'q', 'mq', 't', 'f', 's', 'v',
            'vm', 'vd', 'a', 'j']:
                if word in dict_sum:
                    dict_sum[word] += 1
                else:
                    dict_sum[word] = 1
                    
    # 计算各个实词在语料库占所有词的频率取对数
    for key, value in dict_sum.items():
        dict_sum_frelog[key] = math.log(value / word_sum)
        
    # 统计各个文本中实词频对数平均
    for text in text_pos:
        count_word = 1e-5
        count = 0
        for word, tag in text.items():
            if tag in ['n', 'np', 'ns', 'ni', 'nz', 'm', 'q', 'mq', 't', 'f', 's', 'v',
            'vm', 'vd', 'a', 'j']:
                count += dict_sum_frelog[word]
                count_word += 1
        RWFLA_list.append(count / count_word)

    return RWFLA_list


data_df['RWFLA'] = RWFLA(data_df.cutword, data_df.pos)


# 北大CCL现代汉语字符表 前3000个   和    语料库在线”网站古代汉语语料库字频表 前3000个
common_word_df = pd.read_csv(
    r'data\common_word.csv',
    header=None,
    names=['汉字'])


# 难词数
def difficult_char(text_data):
    text_data = ''.join(text_data.split(' '))
    text_data = set(text_data)
    count = 0
    for i in text_data:
        if i not in common_word_df['汉字'].tolist():
            count += 1
    return count


data_df['difficult_char_num'] = data_df.cutword.apply(difficult_char)


# ### 词汇长度


# 统计文本中笔画字元数及平均笔画数
def text_stroke(text_data):
    low_list = []
    medium_list = []
    high_list = []
    stroke_sum_list = []
    for text in tqdm(text_data):
        text = set(list(''.join(text.split(' '))))
        low = 0
        medium = 0
        high = 0
        stroke_sum = 0
        i = 0  # 统计异常符号数量
        for char in text:
            try:
                temp = stroke_dict[char]
            except KeyError:
                i += 1
                continue
            else:
                stroke_sum += temp
                if temp <= np.quantile(list(stroke_dict.values()), 0.25):
                    low += 1
                elif temp <= np.quantile(list(stroke_dict.values()), 0.75):
                    medium += 1
                else:
                    high += 1
        low_list.append(low)
        medium_list.append(medium)
        high_list.append(high)
        stroke_sum_list.append(stroke_sum / (len(text) - i))
            
    return low_list, medium_list, high_list, stroke_sum_list


data_df['low_stroke'], data_df['medium_stroke'], data_df['high_stroke'], data_df['average_stroke'] = text_stroke(data_df.cutword)


# 二字词数及三字以上词数
def multi_word(text_data):
    two_list = []
    three_list = []
    for text in text_data:
        text = text.split(' ')
        two = 0
        three = 0
        for char in text:
            if len(char) == 2:
                two += 1
            if len(char) >= 3:
                three += 1
        two_list.append(two)
        three_list.append(three)
    return two_list, three_list


data_df['two_word'], data_df['three_word'] = multi_word(data_df.cutword)


# ### 语义类指标

# 实词数  a 形容词,b 其他名词修饰语,j 缩写,m 数词,n 一般名词,nd 方向名词,nh 人名,ni 组织名称,nl 位置名词,ns 地理名称,nt 时态名词,nz 其他专有名词,q 量词,v 动词
def real_word_num(text_data):
    pos = text_data.split(' ')
    real_words = [
        tag for tag in pos if tag in [
             'n', 'np', 'ns', 'ni', 'nz', 'm', 'q', 'mq', 't', 'f', 's', 'v',
            'vm', 'vd', 'a', 'j'
        ]
    ]
    return len(real_words)

data_df['real_word_num'] = data_df.pos.apply(real_word_num)


# ### 句法类指标


# 获取所有标点符号
def punctuation_mark(text_data):
    mark_list = []
    for text in text_data:
        for i in text:
            if i not in stroke_dict and i not in mark_list:
                mark_list.append(i)
    return mark_list



mark_list = punctuation_mark(data_df.text)
print(mark_list)



reserve_mask = ['。','，','？','！',',','；','?','!','…','．','.','—', ';']
abnormal_mask = [i for i in mark_list if i not in reserve_mask]


# 分句 计算简单句和复杂句数量
def split_text(text_data):
    simple_sentence = []
    complex_sentence = []
    for text in text_data:
        text = ''.join([i.strip() for i in text if i not in abnormal_mask])
        # 计算各个文本简单句数量
        temp1 = re.split("[。|，|？|！|,|；|?|!|…|．|.|—|;]", text)
        temp1 = [j for j in temp1 if j != '']
        simple_sentence.append(len(temp1))
        # 计算各个文本复杂句数量
        temp2 = re.split("[。|？|！|?|!|…|．|.|—]", text)
        temp2 = [k for k in temp2 if k != '']
        complex_sentence.append(len(temp2))
    return simple_sentence, complex_sentence


simple_sentence, complex_sentence = split_text(data_df.text)


# 简单句和复杂句的句平均词数
def sentence_avg_word(word_num, simple_sentence, complex_sentence):
    sim_avg_word = []
    com_avg_word = []
    for i, j, k in list(zip(word_num.values, simple_sentence, complex_sentence)):
        sim_avg_word.append(i / j)
        com_avg_word.append(i / k)
    return sim_avg_word, com_avg_word


data_df['sim_avg_word'], data_df['com_avg_word'] = sentence_avg_word(data_df.word_num, simple_sentence, complex_sentence)


# 以。！？…结尾的单句数比率
def sin_sent_ratio(text_data):
    singular_sentence = []
    sentence = []
    ratio = []
    for text in text_data:
        text = ''.join([i.strip() for i in text if i not in abnormal_mask])
        # 计算各个文本句子数
        temp1 = re.split("[。|？|！|?|!|…|．|.|—]", text)
        temp1 = [k for k in temp1 if k != '']
        sentence.append(len(temp1))
        temp2 = []
        for i in temp1:
            n = 0
            for j in ['，', ',', '；', ';']:
                if j in i:
                    break
                else:
                    n += 1
            if n == 4:
                temp2.append(i)
        singular_sentence.append(len(temp2))
    for a, b in zip(singular_sentence, sentence):
        ratio.append(a / b)
    return ratio


data_df['sin_sent_ratio'] = sin_sent_ratio(data_df.text)


mark_list = punctuation_mark(data_df.text)
reserve_mask = ['。','，','？','！',',','；','?','!','…','．','.','—', ';','：','、', '《', '》', ':']
abnormal_mask = [i for i in mark_list if i not in reserve_mask]


# 名词片语比率
def nsubj_ratio(text_data, abnormal_mask):
    HanLP = hanlp.pipeline() \
    .append(hanlp.utils.rules.split_sentence, output_key='sentences') \
    .append(hanlp.load('FINE_ELECTRA_SMALL_ZH'), output_key='tok') \
    .append(hanlp.load('CTB9_DEP_ELECTRA_SMALL'), output_key='dep', input_key='tok')
    ratio_list = []
    for text in tqdm(range(len(text_data))):
        # 清洗文本
        text_str = ''
        for i in text_data.iloc[text].text:
            if i in stroke_dict or  i not in abnormal_mask:
                text_str += i
        # 获取名词片语
        doc=HanLP(text_str)
        doc_list = list(chain.from_iterable(doc['dep']))
        nsubj_num = 0
        for i in doc_list:
            if i['deprel'] == 'nsubj':
                nsubj_num += 1
        ratio_list.append(nsubj_num / len(doc_list))
    return ratio_list


data_df['nsubj_ratio'] = nsubj_ratio(data_df, abnormal_mask)


# ## 文章凝聚性类指标
# ### 指称词

# 代词数
def pronoun_num(text_data):
    pos = text_data.split(' ')
    pronoun_words = [
        tag for tag in pos if tag == 'r'
    ]
    return len(pronoun_words)


data_df['pronoun_num'] = data_df.pos.apply(pronoun_num)


# ### 连接词


# 连接词数
def conjunction_num(text_data, pos_data):
    tag_num = []
    conjunction_list = []
    for i in tqdm(range(len(text_data))):
        text = text_data.iloc[i].split(' ')
        pos = pos_data.iloc[i].split(' ')
        tag_list = []
        for word, tag in zip(text, pos):
            # 找出未清洗干净的文本
            if tag == 'c' and len(word) > 20:
                print(i)
            elif tag == 'c' and len(word) < 20:
                tag_list.append(tag)
                conjunction_list.append(word)
        tag_num.append(len(tag_list))
    return tag_num, set(conjunction_list)


data_df['conjunction_num'], conjunction_set = conjunction_num(data_df.cutword, data_df.pos)


# 正负向连接词数
pos_conjunction = ['于是', '假如', '果真', '如若', '甚而', '同时', '如', '如果', '与', '进而', '换言之', '然后', '从而', '就是说', 
                  '就', '以及', '此外', '或者', '这就是说', '尔后', '因为', '和', '只要', '无论', '甚至', '所以', '甚或', '相对而言', 
                   '不但', '换句话说', '由此可见', '继而', '无论是', '紧接着', '并且', '何况', '之所以', '故', '不仅', '才', '就是', 
                  '以致', '因', '以免', '倘然', '接着', '既', '由于', '并', '不论', '诚然', '总之', '既然如此', '因而', '其次', '首先',
                  '果然', '可见', '或', '要是说', '若是', '若', '只是', '倘若', '因此', '既然', '再者', '与此同时', '不管', '或是', 
                  '况且', '倘使', '再其次', '总而言之', '假若', '一方面', '再说', '而且', '且', '一旦', '即使', '假说', '要是', '论是', 
                   '另一方面', '故此', '乃至']
neg_conjunction = ['可', '固然', '则', '但是', '虽', '还是', '宁愿', '纵使', '忽而', '虽然', '以至于', '与其', '转而', '然', '但', 
                   '否', '而', '除非', '尽管', '毋宁', '可是', '虽说','反之', '不过', '非但', '但凡', '纵然', '另外', '万一', 
                   '恰恰相反', '以至', '而是', '只有', '以防', '即便', '而况', '否则', '否则的', '与其说', '然而', '——否则', '陡然', ]
def pos_neg_conjunction(text_data, pos_data, pos_conjunction, neg_conjunction):
    tag_pos = []
    tag_neg = []
    for i in tqdm(range(len(text_data))):
        text = text_data.iloc[i].split(' ')
        pos = pos_data.iloc[i].split(' ')
        pos_list = []
        neg_list = []
        for word, tag in zip(text, pos):
            if tag == 'c' and word in pos_conjunction:
                pos_list.append(tag)
            if tag == 'c' and word in neg_conjunction:
                neg_list.append(tag)
        tag_pos.append(len(pos_list))
        tag_neg.append(len(neg_list))
    return tag_pos, tag_neg


data_df['pos_conjunction'], data_df['neg_conjunction'] = pos_neg_conjunction(data_df.cutword, data_df.pos, pos_conjunction, neg_conjunction)


# # Exploring the Impact of Linguistic Features for Chinese Readability Assessment
# ## Discourse features
# ### Entity density


abnormal_mask2 = ['─', '─', '．', '．', '∶', '×', '．', '·', '─', '①', '─', 'ー', '∶', '∶', '—', 'è', 'ɡ', 'à', '—', 
'ɡ', 'à', 'é', '∶', '．', '①', '∶', 'è', '．', 'à', 'í', '①', 'í', '┅', '℃', '．', '∶', '×', '①', '①', '①', '～', '•', 
'②', '•', '〔', '〕', '⑹', '⑺', '⑷', '⑶', '—', '⑸', '⑵', '•', '─', 'è', 'ù', 'ó', 'ú', 'í', '•', '～', '×', '―', '〔', 
'○', 'ī', 'ǔ', '〇', '～', '─', '─', '〕', '-', '⑹', '⑺', '⑷', '⑶', '—', '⑸', '⑵', '①', '①', '②', '─', '⑥', '．', 
'·', '⑩', '⑾', '-', '∶', '-', '．', '―', '―', '∶', '⑹', '⑺', '⑷', '⑶', '—', '⑸', '⑵', '•', '①', '②', '○', '•', 
'•', '⑦', '⑩', '⑾', '⑵', '-', '③', '①', '②', '．', '°', 'α', '─', '．', '．', '─', '．', '％', '℃', '②', '∶', '℃', 
'%', 'Ⅱ', 'げ', '•', 'α', '─', '〔', '①', '─', 'ā', '〕', '①', '—', '．', '－', '·', '·', '·', 'í', '·', 'á', '﹖', 
'í', 'Ｃ', 'è', '～', '．', '□', '─', '－', '／', 'Ｆ', '〔', '③', '〔', '〕', '─', 'П', '③', '．', '②', '⑥', '•', '⑤', '④']


# 计算命名实体特征
def named_entity(text_data, abnormal_mask):
    entity_num = []
    uni_entity_num = []
    entity_percentage = []
    uni_entity_percentage = []
    avg_sent_entity = []
    avg_sent_uni_entity = []
    HanLP_ner = hanlp.pipeline() \
    .append(hanlp.utils.rules.split_sentence, output_key='sentences') \
    .append(hanlp.load('FINE_ELECTRA_SMALL_ZH'), output_key='tok') \
    .append(hanlp.load('MSRA_NER_ELECTRA_SMALL_ZH'), output_key='ner', input_key='tok') 
    for text in tqdm(range(len(text_data))):
        entity_list = []
        # 清洗文本
        text_str = ''
        for i in text_data.iloc[text].text:
            if i not in abnormal_mask:
                text_str += i
        # 获取命名实体
        doc=HanLP_ner(text_str)
        doc_list = list(chain.from_iterable(doc['ner']))
        for j in doc_list:
            entity_list.append(j[0])
        entity_set = set(entity_list)
        # 获取文本词数
        word_list = text_data.iloc[text].cutword.split(' ')
        word_set = set(word_list)
        # 计算命名实体数
        entity_num.append(len(entity_list))
        # 计算唯一命名实体数
        uni_entity_num.append(len(entity_set))
        # 计算命名实体百分比
        entity_percentage.append(len(entity_list) / len(word_list))
        # 计算唯一命名实体百分比
        uni_entity_percentage.append(len(entity_set) / len(word_set))
        # 计算平均每句命名实体数
        avg_sent_entity.append(len(entity_list) / complex_sentence[text])
        # 计算平均每句唯一命名实体数
        avg_sent_uni_entity.append(len(entity_set) / complex_sentence[text])
    return entity_num, uni_entity_num, entity_percentage, uni_entity_percentage, avg_sent_entity, avg_sent_uni_entity



data_df['entity_num'], data_df['uni_entity_num'], data_df[
    'entity_percentage'], data_df['uni_entity_percentage'], data_df[
        'avg_sent_entity'], data_df['avg_sent_uni_entity'] = named_entity(data_df, abnormal_mask2)


# ## POS features
# ### Adjectives, Nouns, Verbs


def pos_feat(text_df, complex_sentence):
    # 形容词
    adj_percentage = []; uni_adj_percentage = [];  uni_adj_num = []; avg_sent_adj = []; avg_sent_uni_adj = []
    # 名词
    n_percentage = []; uni_n_percentage = [];  uni_n_num = []; avg_sent_n = []; avg_sent_uni_n = []
    # 动词
    v_percentage = []; uni_v_percentage = [];  uni_v_num = []; avg_sent_v = []; avg_sent_uni_v = []
    
    for text in tqdm(range(len(text_df))):
        text_data = text_df.iloc[text].cutword
        text_data = text_data.split(' ')
        pos_data = text_df.iloc[text].pos
        pos_data = pos_data.split(' ')
        adj_list = []  # 形容词
        n_list = []  # 名词
        v_list = []   # 动词
        for word, tag in zip(text_data, pos_data):
            if tag == 'a':
                adj_list.append(word)
            if tag == 'n' or tag == 'np' or tag == 'ns' or tag == 'ni' or tag == 'nz':
                n_list.append(word)
            if tag == 'v':
                v_list.append(word)
        
        # 形容词特征
        ## adj_percentage：文本形容词百分比，uni_adj_percentage：文本唯一形容词百分比，uni_adj_num：文本唯一形容词数量
        ## avg_sent_adj：平均每句形容词数量，avg_sent_uni_adj：平均每句唯一形容词数量
        adj_percentage.append(len(adj_list) / len(text_data))
        uni_adj_percentage.append(len(set(adj_list)) / len(set(text_data)))
        uni_adj_num.append(len(set(adj_list)))
        avg_sent_adj.append(len(adj_list) / complex_sentence[text])
        avg_sent_uni_adj.append(len(set(adj_list)) / complex_sentence[text])
        
        # 名词特征
        ## n_percentage：文本名词百分比，uni_n_percentage：文本唯一名词百分比，uni_n_num：文本唯一名词数量
        ## avg_sent_n：平均每句名词数量，avg_sent_uni_n：平均每句唯一名词数量
        n_percentage.append(len(n_list) / len(text_data))
        uni_n_percentage.append(len(set(n_list)) / len(set(text_data)))
        uni_n_num.append(len(set(n_list)))
        avg_sent_n.append(len(n_list) / complex_sentence[text])
        avg_sent_uni_n.append(len(set(n_list)) / complex_sentence[text])
        
        # 动词特征
        ## v_percentage：文本动词百分比，uni_v_percentage：文本唯一动词百分比，uni_v_num：文本唯一动词数量
        ## avg_sent_v：平均每句动词数量，avg_sent_uni_v：平均每句唯一动词数量
        v_percentage.append(len(v_list) / len(text_data))
        uni_v_percentage.append(len(set(v_list)) / len(set(text_data)))
        uni_v_num.append(len(set(v_list)))
        avg_sent_v.append(len(v_list) / complex_sentence[text])
        avg_sent_uni_v.append(len(set(v_list)) / complex_sentence[text])
        
    return adj_percentage, uni_adj_percentage,  uni_adj_num, avg_sent_adj, avg_sent_uni_adj, \
    n_percentage, uni_n_percentage,  uni_n_num, avg_sent_n, avg_sent_uni_n, \
    v_percentage, uni_v_percentage,  uni_v_num, avg_sent_v, avg_sent_uni_v


data_df['adj_percentage'], data_df['uni_adj_percentage'], data_df[
    'uni_adj_num'], data_df['avg_sent_adj'], data_df[
        'avg_sent_uni_adj'], data_df['n_percentage'], data_df[
            'uni_n_percentage'], data_df['uni_n_num'], data_df[
                'avg_sent_n'], data_df['avg_sent_uni_n'], data_df[
                    'v_percentage'], data_df['uni_v_percentage'], data_df[
                        'uni_v_num'], data_df['avg_sent_v'], data_df[
                            'avg_sent_uni_v'] = pos_feat(data_df, complex_sentence)


# # Research on the Evaluation of the Classical Chinese Difficulty in the Compulsory Education Stage
# ## Word features of classical Chinese
# ### Lexical Density



# 计算虚词数量和密度
def function_words(text_data):
    function_num = []
    function_density = []
    for text in text_data:
        text_list = text.split(' ')
        count = 0
        for word in text_list:
            if word in ['d', 'c', 'p', 'u', 'y', 'e', 'o']:
                count += 1
        function_num.append(count)
        function_density.append(count / len(text_list))
    return function_num, function_density


data_df['function_num'], data_df['function_density'] = function_words(data_df.pos,)

# ### Diversity of Words


# RTTR
def RTTR(text_data):
    text_list = text_data.split(' ')
    text_set = set(text_list)
    return len(text_set) / math.sqrt(len(text_list))


data_df['RTTR'] = data_df.cutword.apply(RTTR)

# 参考 https://ntcuir.ntcu.edu.tw/bitstream/987654321/867/1/102NTCT0629015-001.pdf
# MTLD
def calculate_ttr(tokens):
    unique_tokens = set(tokens)
    ttr = len(unique_tokens) / len(tokens)
    return ttr

def MTLD(text_data, threshold=0.72):
    text_list = text_data.split(' ')
    start = 0
    factor_num = 0   # 因子个数
    for stop in range(len(text_list)):
        text = text_list[start:stop+1]
        ttr = calculate_ttr(text)
        if ttr < threshold:
            start = stop+1
            if len(text) >= 5:
                factor_num += 1
    if len(text) > 5 and ttr > threshold:
        factor_num += 1
    # RS 指剩餘的分數：文本分析結束點的 TTR 值。
    RS = ttr
    # FS 指因子得分，在這項分析中被設定為 0.72。 
    FS = 0.72
    # IFS 指被加入到因子數的不完全因子得分。
    IFS = (1 - RS) / (1 - FS)
    # Measure of Textual Lexical Diversity
    mtld = len(text_list) / (factor_num + IFS)
    return mtld

def reverse_string(string):
    reversed_string = string[::-1]
    return reversed_string

def calculate_MTLD(text_data):
    # 顺序
    mtld1 = MTLD(text_data)
    # 逆序
    mtld2 = MTLD(reverse_string(text_data))
    # MTLD
    mtld = (mtld1 + mtld2) / 2
    return mtld

data_df['MTLD'] = data_df.cutword.apply(calculate_MTLD)


data_df.to_csv(
    r'data_collect\CLT\temp.csv',
    index=False,
    encoding='utf-8')