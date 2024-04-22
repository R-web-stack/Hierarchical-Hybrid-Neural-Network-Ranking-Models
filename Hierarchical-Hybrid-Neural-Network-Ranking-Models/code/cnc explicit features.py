#!/usr/bin/env python
# coding: utf-8


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import re
import math
from jiayan import load_lm
from jiayan import CharHMMTokenizer
from jiayan import CRFPOSTagger
import stanza
import udkanbun
from tqdm import tqdm


from io import StringIO

from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser


data_df = pd.read_csv(r'data_collect\CMCC\text_sum.txt',sep='\t',
           header=None, names = ['label', 'text'], encoding='utf-8')


lm = load_lm(r'D:\anaconda\Lib\site-packages\jiayan\jiayan_models\jiayan.klm')
tokenizer = CharHMMTokenizer(lm)
stop_words = pd.read_csv(
    r"data\stopwords-master\classical_stopwords.txt",
    header=None,
    names=["text"],
    sep="\r",
    quoting=3)


## Preprocess Chinese text data by removing unnecessary characters, segmenting words, removing stop words, and other operations
def chinese_pre(text_data):
    ## Remove letters, numbers, and spaces
    text_data = re.sub('[a-zA-Z]', '', text_data)
    text_data = re.sub("\d+", "", text_data)
    text_data = re.sub(" ", "", text_data)
    ## Word segmentation, using precise patterns
    text_data = list(tokenizer.tokenize(text_data))
    ## Remove stop words and extra spaces
    text_data = [word.strip() for word in text_data if word not in stop_words.text.values]
    ## The processed words are concatenated into strings using spaces
    text_data = " ".join(text_data)
    return text_data


data_df["cutword"] = data_df.text.apply(chinese_pre)


# # Investigating Chinese Text Readability: Linguistic Features, Modeling, and Validation
# ## 词汇类指标
# ### 词汇数量


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


# 实词密度  a 形容词,b 其他名词修饰语,j 缩写,m 数词,n 一般名词,nd 方向名词,nh 人名,ni 组织名称,nl 位置名词,ns 地理名称,nt 时态名词,nz 其他专有名词,q 量词,v 动词
def real_word(text_data):
    text_data = text_data.split(' ')
    postagger = CRFPOSTagger()
    postagger.load(
        r'D:\anaconda\Lib\site-packages\jiayan\jiayan_models\pos_model')
    pos = postagger.postag(text_data)
    real_words = [
        tag for tag in pos if tag in [
            'a', 'b', 'j', 'm', 'n', 'nd', 'nh', 'ni', 'nl', 'ns', 'nt', 'nz',
            'q', 'v'
        ]
    ]
    return len(real_words) / len(text_data)


data_df['real_word'] = data_df.cutword.apply(real_word)


# ### 词汇频率


# 实词频对数平均(Real Word Frequency Logarithmic Average)  a 形容词,j 缩写,n 一般名词,ni 组织名称,nz 其他专有名词,v 动词
def RWFLA(text_data):
    dict_sum = {}  # 各个实词在语料库的总数表
    dict_sum_frelog = {}   # 各个实词在语料库占所有词的频率取对数表
    word_sum = 0   # 语料库词总数
    text_pos = []  # 文本词性标注配对表
    RWFLA_list = []   # 各个文本中实词频对数平均表
    
    # 对所有文本进行词性标注
    for i in text_data:
        text = i.split(' ')
        word_sum += len(text)
        postagger = CRFPOSTagger()
        postagger.load(
            r'D:\anaconda\Lib\site-packages\jiayan\jiayan_models\pos_model')
        pos = postagger.postag(text)
        text_pos.append(dict(zip(text, pos)))
        # 统计各个文本中较有意义的实词
        for word, tag in zip(text, pos):
            if tag in ['a', 'j', 'n', 'ni', 'nz', 'v']:
                if word in dict_sum:
                    dict_sum[word] += 1
                else:
                    dict_sum[word] = 1
                    
    # 计算各个实词在语料库占所有词的频率取对数
    for key, value in dict_sum.items():
        dict_sum_frelog[key] = math.log(value / word_sum)
        
    # 统计各个文本中实词频对数平均
    for text in text_pos:
        count_word = 0
        count = 0
        for word, tag in text.items():
            if tag in ['a', 'j', 'n', 'ni', 'nz', 'v']:
                count += dict_sum_frelog[word]
                count_word += 1
        RWFLA_list.append(count / count_word)

    return RWFLA_list


data_df['RWFLA'] = RWFLA(data_df.cutword)


# 提取北大CCL古代汉语字符表 前3000个
output_string = StringIO()
with open(r'data\gudai_char_info.pdf', 'rb') as in_file:
    parser = PDFParser(in_file)
    doc = PDFDocument(parser)
    rsrcmgr = PDFResourceManager()
    device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    for page in PDFPage.create_pages(doc):
        interpreter.process_page(page)

common_word1 = re.sub(
    ":", "",
    re.sub(
        "\d+", "",
        re.sub(
            '\x0c', '',
            re.sub('\n', '',
                   re.sub('\xa0', '',
                          output_string.getvalue()[2613:126638])))))[:3000]
common_word1 = list(common_word1)

# 提取“语料库在线”网站古代汉语语料库字频表 前3000个
sheet = pd.read_excel(r'data\ACCorpusCharacterlist.xls',header = 0,skiprows= 6)
common_word2 = sheet['汉字'][:3000].tolist()

# 合并
common_word = set(common_word1 + common_word2)



pd.DataFrame(common_word).to_csv(
    r'data_collect\CMCC\common_word.csv',
    index=False,
    header=False)



common_word_df = pd.read_csv(
    r'data_collect\CMCC\common_word.csv',
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


# 笔画函数
def get_stroke(c):
    # 如果返回 0, 则也是在unicode中不存在kTotalStrokes字段
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

# 构建笔画数字表
def stroke_table(text_data):
    stroke_dict = {}
    count = 0
    for text in text_data:
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
    r'data_collect\CMMC\char_stroke.xlsx',
    index=False,
    header=False)


char_stroke = pd.read_excel(r'data_collect\CMCC\char_stroke.xlsx', header=None)


stroke_dict = {}
for i in range(len(char_stroke)):
    stroke_dict[char_stroke.iloc[i][0]] = char_stroke.iloc[i][1]


# 统计文本中笔画字元数及平均笔画数
def text_stroke(text_data):
    low_list = []
    medium_list = []
    high_list = []
    stroke_sum_list = []
    for text in text_data:
        text = set(list(''.join(text.split(' '))))
        low = 0
        medium = 0
        high = 0
        stroke_sum = 0
        for char in text:
            temp = get_stroke(char)
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
        stroke_sum_list.append(stroke_sum / len(text))
            
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
    text_data = text_data.split(' ')
    postagger = CRFPOSTagger()
    postagger.load(
        r'D:\anaconda\Lib\site-packages\jiayan\jiayan_models\pos_model')
    pos = postagger.postag(text_data)
    real_words = [
        tag for tag in pos if tag in [
            'a', 'b', 'j', 'm', 'n', 'nd', 'nh', 'ni', 'nl', 'ns', 'nt', 'nz',
            'q', 'v'
        ]
    ]
    return len(real_words)


data_df['real_word_num'] = data_df.cutword.apply(real_word_num)


stanza.download('lzh')



# 否定词
def neg_word(text_data):
    #  download_method=None 这个参数去掉的话，检查resources.json的更新，以防模型已经更新。
    # verbose=False 去掉打印信息
    nlp = stanza.Pipeline('lzh', download_method=None, processors='tokenize,pos', verbose=False) # 默认设置去掉  processors 参数
    doc = nlp(text_data)
    neg = 0
    for sent in doc.sentences:
        for word in sent.words:
            if '否定' in word.xpos:
                neg += 1
    return neg


data_df['neg_word'] = data_df.text.apply(neg_word)

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
    lzh=udkanbun.load(MeCab=False)
    ratio_list = []
    for text in range(len(text_data)):
        # 清洗文本
        text_str = ''
        for i in text_data.iloc[text].text:
            if i in stroke_dict or  i not in abnormal_mask:
                text_str += i
        # 获取名词片语
        s=lzh(text_str)
        nsubj_num = s.to_tree().count('nsubj')
        ratio_list.append(nsubj_num / len(text_data.iloc[text].cutword.split(' ')))
    return ratio_list


data_df['nsubj_ratio'] = nsubj_ratio(data_df)


# ## 文章凝聚性类指标
# ### 指称词



# 代词数
def pronoun_num(text_data):
    text_data = text_data.split(' ')
    postagger = CRFPOSTagger()
    postagger.load(
        r'D:\anaconda\Lib\site-packages\jiayan\jiayan_models\pos_model')
    pos = postagger.postag(text_data)
    pronoun_words = [
        tag for tag in pos if tag == 'r'
    ]
    return len(pronoun_words)



data_df['pronoun_num'] = data_df.cutword.apply(pronoun_num)


# ### 连接词


# 连接词数
def conjunction_num(text_data):
    postagger = CRFPOSTagger()
    postagger.load(
        r'D:\anaconda\Lib\site-packages\jiayan\jiayan_models\pos_model')
    tag_num = []
    conjunction_list = []
    for text in text_data:
        text = text.split(' ')
        pos = postagger.postag(text)
        tag_list = []
        for word, tag in zip(text, pos):
            if tag == 'c':
                tag_list.append(tag)
                conjunction_list.append(word)
        tag_num.append(len(tag_list))
    return tag_num, set(conjunction_list)



data_df['conjunction_num'], conjunction_set = conjunction_num(data_df.cutword)



# 正负向连接词数
pos_conjunction = ['既克', '则', '因说', '既至', '暨', '多', '及诸', '和', '与天', '与诸', '且至', '因是', '且行且', '由是', '因',
                  '故', '与魏', '与', '与汝', '与李', '因使', '既平', '若', '与王', '与我', '以', '且', '倘', '或', '如是', '既诛',
                  '日', '昇', '并', '若是', '既', '如']
neg_conjunction = ['虽强', '才', '苟', '虽微', '虽死', '纵使', '虽少', '虽欲', '又', '然', '虽', '苟且', '但', '宁', '但当', '然自',
                  '虽云', '与宋', '纵', '虽贵', '否则', '即令', '乃', '与同', '与君', '然后', '因言', '及', '虽无', '然若', '而',
                  '即便']
def pos_neg_conjunction(text_data, pos_conjunction, neg_conjunction):
    postagger = CRFPOSTagger()
    postagger.load(
        r'D:\anaconda\Lib\site-packages\jiayan\jiayan_models\pos_model')
    tag_pos = []
    tag_neg = []
    for text in text_data:
        text = text.split(' ')
        pos = postagger.postag(text)
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


data_df['pos_conjunction'], data_df['neg_conjunction'] = pos_neg_conjunction(data_df.cutword, pos_conjunction, neg_conjunction)


# # Exploring the Impact of Linguistic Features for Chinese Readability Assessment
# ## Discourse features
# ### Entity density


# https://huggingface.co/ethanyt/guwen-ner
'''id2label": {
    "0": "O",
    "1": "B-NOUN_BOOKNAME",
    "2": "I-NOUN_BOOKNAME",
    "3": "B-NOUN_OTHER",
    "4": "I-NOUN_OTHER"
  }
'''
from transformers import AutoTokenizer, AutoModelForTokenClassification
tokenizer = AutoTokenizer.from_pretrained("ethanyt/guwen-ner")
model = AutoModelForTokenClassification.from_pretrained("ethanyt/guwen-ner")


from transformers import pipeline
# aggregation_strategy 参数设置为 None，将输出原始模式，即单个字。参数说明可以看 ?ner
ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="average")


# 获取实体列表
def get_full_words(text, entities):
    word_list = list(text)
    full_words = []
    for entity in entities:
        start = entity['start']
        end = entity['end']
        full_word = ''.join(word_list[start:end])
        full_words.append(full_word)
    return full_words


# 计算命名实体特征
def named_entity(text_data, ner, abnormal_mask, get_full_words, complex_sentence):
    entity_num = []
    uni_entity_num = []
    entity_percentage = []
    uni_entity_percentage = []
    avg_sent_entity = []
    avg_sent_uni_entity = []
    for text in range(len(text_data)):
        # 清洗文本
        text_str = ''
        for i in text_data.iloc[text].text:
            if i in stroke_dict or  i not in abnormal_mask:
                text_str += i
        # 分成简单句分析命名实体
        temp1 = re.split("[。|，|？|！|,|；|?|!|…|．|.|—|;]", text_str)
        temp1 = [j for j in temp1 if j != '']
        entity_list = []
        for sim_text in temp1:
            try:
                # 获取命名实体
                result = ner(sim_text)
            except IndexError:
                pass
            else:
                entity_list.extend(get_full_words(text_str, result))
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
        'avg_sent_entity'], data_df['avg_sent_uni_entity'] = named_entity(
            data_df, ner, abnormal_mask, get_full_words, complex_sentence)


# ## POS features
# ### Adjectives, Nouns, Verbs


def pos_feat(text_df, complex_sentence):
    postagger = CRFPOSTagger()
    postagger.load(
        r'D:\anaconda\Lib\site-packages\jiayan\jiayan_models\pos_model')
    # 形容词
    adj_percentage = []; uni_adj_percentage = [];  uni_adj_num = []; avg_sent_adj = []; avg_sent_uni_adj = []
    # 名词
    n_percentage = []; uni_n_percentage = [];  uni_n_num = []; avg_sent_n = []; avg_sent_uni_n = []
    # 动词
    v_percentage = []; uni_v_percentage = [];  uni_v_num = []; avg_sent_v = []; avg_sent_uni_v = []
    
    for text in range(len(text_df)):
        text_data = text_df.iloc[text].cutword
        text_data = text_data.split(' ')
        pos = postagger.postag(text_data)
        adj_list = []  # 形容词
        n_list = []  # 名词
        v_list = []   # 动词
        for word, tag in zip(text_data, pos):
            if tag == 'a':
                adj_list.append(word)
            if tag == 'n':
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


# 常用虚词表
function_words_df = pd.read_csv(
    r'data\虚词.txt',
    header=None,
    names=['汉字'])


# 计算常用虚词数量和密度
def function_words(text_data, function_words_df):
    function_num = []
    function_density = []
    for text in text_data:
        text_list = text.split(' ')
        count = 0
        for word in text_list:
            if word in function_words_df['汉字'].values:
                count += 1
        function_num.append(count)
        function_density.append(count / len(text_list))
    return function_num, function_density


data_df['function_num'], data_df['function_density'] = function_words(data_df.cutword, function_words_df)


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


# # self-features
# ## Text features of classical Chinese
# ### 藏书类别
# 在古文中，"易藏"、"医藏"、"艺藏"、"史藏"、"佛藏"、"集藏"、"诗藏"、"子藏"、"儒藏"和"道藏"是指不同的文献或书籍收藏类别，每个类别都有其特定的内容和用途。它们的主要区别如下：  
# 
# 易藏：指的是《易经》及其相关的文献和注释，包括《易传》、《十翼》等。  
# 
# 医藏：指的是古代医学著作的集合，包括各种医书、医经和医方等。  
# 
# 艺藏：指的是艺术相关的文献和书籍，包括音乐、绘画、舞蹈、戏剧等各种艺术形式的著作和记录。  
# 
# 史藏：指的是历史著作和历史文献的集合，包括各种历史记载、史书、编年体著作等。  
# 
# 佛藏：指的是佛教经典和佛教文献的集合，包括佛经、佛教教义著作、宗派经典等。  
# 
# 集藏：指的是各种不同类别的文献和书籍的综合收藏，包括各个学科领域的文献和著作。  
# 
# 诗藏：指的是诗歌作品的集合，包括各个历史时期和流派的诗人的作品集。  
# 
# 子藏：指的是儒家经典和儒家文献的集合，包括《论语》、《孟子》、《大学》等经典著作。  
# 
# 儒藏：指的是儒家经典和儒家文献的整理和收集，常用来指儒家的典籍和学说体系。  
# 
# 道藏：指的是道家经典和道家文献的集合，包括《道德经》、《庄子》、《列子》等道家的经典著作。  
# 
# 这些不同的藏书类别代表了古代不同领域的学问和知识体系，每个类别都有其独特的特点和价值。它们在古代文化和学术领域具有重要的地位，对于研究和了解古代文化、思想和知识有着重要的作用。  


# https://huggingface.co/ethanyt/guwen-cls
# 藏书类别 
'''id2label": {
    "0": "易藏",
    "1": "医藏",
    "2": "艺藏",
    "3": "史藏",
    "4": "佛藏",
    "5": "集藏",
    "6": "诗藏",
    "7": "子藏",
    "8": "儒藏",
    "9": "道藏"
  }'''
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer_cls = AutoTokenizer.from_pretrained("ethanyt/guwen-cls")
model_cls = AutoModelForSequenceClassification.from_pretrained("ethanyt/guwen-cls")


mark_list = punctuation_mark(data_df.text)
reserve_mask = ['。','，','？','！',',','；','?','!','…','．','.','—', ';','：','、', '《', '》', ':']
abnormal_mask = [i for i in mark_list if i not in reserve_mask]

def library_category(text_data):
    # 清洗文本
    text_str = ''
    for i in text_data:
        if i in stroke_dict or  i not in abnormal_mask:
            text_str += i
    input_ids = tokenizer_cls.encode(text_str, truncation=True, padding=True, max_length=512, return_tensors="pt")
    outputs = model_cls(input_ids)
    predictions = outputs.logits.argmax(dim=1)
    predicted_label = predictions.item()
    return predicted_label


data_df['library_category'] = data_df.text.apply(library_category)


# ### 情感类别


# https://huggingface.co/ethanyt/guwen-sent
# 情感类别
'''id2label": {
    "0": "Neg",
    "1": "ImpNeg",
    "2": "Nerual",
    "3": "ImpPos",
    "4": "Pos"
  }
'''
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer_sent = AutoTokenizer.from_pretrained("ethanyt/guwen-sent")
model_sent = AutoModelForSequenceClassification.from_pretrained("ethanyt/guwen-sent")


def emotional_category(text_data):
    # 清洗文本
    text_str = ''
    for i in text_data:
        if i in stroke_dict or  i not in abnormal_mask:
            text_str += i
    input_ids = tokenizer_sent.encode(text_str, truncation=True, padding=True, max_length=512, return_tensors="pt")
    outputs = model_sent(input_ids)
    predictions = outputs.logits.argmax(dim=1)
    predicted_label = predictions.item()
    return predicted_label


data_df['emotional_category'] = data_df.text.apply(emotional_category)


# ## 复杂语义
# ### 通假字


# 常用通假字表
interchangeable_words_df = pd.read_csv(
    r'data\通假字.txt',
    header=None,
    sep = '\t：',
    names=['汉字', '释义'])


def interchangeable_words(text_data, interchangeable_words_df):
    count_list = []
    for text in range(len(data_df)):
        char_list = text_data.iloc[text].text
        word_list = [word for word in text_data.iloc[text].cutword.split(' ') if len(word) > 1]
        count = 0
        for char in char_list:
            if char in interchangeable_words_df['汉字'].values:
                count += 1
        for word in word_list:
            if word in interchangeable_words_df['汉字'].values:
                count += 1
        count_list.append(count)
    return count_list


data_df['interchangeable_words'] = interchangeable_words(data_df, interchangeable_words_df)


# ### 古今异义

# In[11]:


# 古今异义表  Ancient and Modern Synonyms
AMS_df = pd.read_csv(
    r'data\古今异义.txt',
    header=None,
    sep = '\t：',
    names=['汉字', '释义'])


# Ancient and Modern Synonyms
def AMS(text_data, AMS_df):
    count_list = []
    for text in range(len(data_df)):
        char_list = text_data.iloc[text].text
        word_list = [word for word in text_data.iloc[text].cutword.split(' ') if len(word) > 1]
        count = 0
        for char in char_list:
            if char in AMS_df['汉字'].values:
                count += 1
        for word in word_list:
            if word in AMS_df['汉字'].values:
                count += 1
        count_list.append(count)
    return count_list


data_df['AMS'] = AMS(data_df, AMS_df)

# ### 一词多义


# 一词多义表
polysemy_df = pd.read_csv(
    r'data\一词多义.txt',
    header=None,
    sep = '\t：',
    names=['汉字', '释义'])


def polysemy(text_data, polysemy_df):
    count_list = []
    for text in range(len(data_df)):
        char_list = text_data.iloc[text].text
        word_list = [word for word in text_data.iloc[text].cutword.split(' ') if len(word) > 1]
        count = 0
        for char in char_list:
            if char in polysemy_df['汉字'].values:
                count += 1
        for word in word_list:
            if word in polysemy_df['汉字'].values:
                count += 1
        count_list.append(count)
    return count_list


data_df['polysemy'] = polysemy(data_df, polysemy_df)

data_df.to_csv(
    r'data_collect\CMCC\features.csv',
    index=False,
    encoding='utf-8')