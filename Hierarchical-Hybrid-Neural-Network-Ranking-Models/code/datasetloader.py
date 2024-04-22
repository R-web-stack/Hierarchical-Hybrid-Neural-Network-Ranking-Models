# coding=utf-8

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from d2l import torch as d2l
import collections
import torch
from torch.utils import data

def split_sentences(text_data, separator):
    """Split text lines into sentences"""
    text_list = []
    for text in text_data:
        sent_list = text.split(separator)
        text_list.append(sent_list)
    return text_list
    
    
def tokenize(lines, token='word'):
    """Split text lines into words or character elements"""
    if token == 'word':
        lines_list = []
        for line in lines:
            word_list = line.split()
            lines_list.append(word_list)
        return lines_list
    elif token == 'char':
        lines_list = []
        for line in lines:
            char_list = list(line)
            lines_list.append(char_list)
        return lines_list
    else:
        print('Error: Unknown word type:' + token)
        
        
class Vocab:
    """Text vocabulary"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # Sort by frequency of occurrence
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # The index of unknown word element is 0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):   # Get the index of word elements
        if not isinstance(tokens, (list, tuple)):    # If the input is not a list or tuple, but a single word element
            return self.token_to_idx.get(tokens, self.unk)    # Return the corresponding index. If it does not exist, return the unknown word index
        return [self.__getitem__(token) for token in tokens]   # If a list of words is passed in, return the corresponding index list

    def to_tokens(self, indices):   # Get the word elements corresponding to the index
        if not isinstance(indices, (list, tuple)):  # If the input is not a list or tuple, but a single index value
            return self.idx_to_token[indices]    # Return the corresponding word element
        return [self.idx_to_token[index] for index in indices]    # If an index list is passed in, return the corresponding list of word elements

    @property
    def unk(self):  # The index of unknown word element is 0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):
    """The frequency of statistical word elements"""
    # If it is a hierarchical text - sentences, words
    if isinstance(tokens[0][0], list):
        # Flatten the list of word elements into a list
        tokens = [token for text in tokens for line in text for token in line]
    # The tokens here are either 1D lists or 2D lists
    elif len(tokens) == 0 or isinstance(tokens[0][0], list) or isinstance(tokens[0], list):
        # Flatten the list of word elements into a list
        tokens = [token for line in tokens for token in line]
    
    return collections.Counter(tokens)
    
    
def sent_truncate_pad(text, num_steps, sent_max_len, padding_token):
    """Truncate or fill in text sentences"""
    valid_len = len(text)
    if valid_len > num_steps:
        return text[:num_steps], num_steps
    return text + [[padding_token] * sent_max_len] * (num_steps - len(text)), valid_len
    
    
def truncate_pad(line, num_steps, padding_token):
    """Truncate or fill text sequences"""
    valid_len = len(line)
    if valid_len > num_steps:
        return line[:num_steps], num_steps  # truncate
    return line + [padding_token] * (num_steps - len(line)), valid_len  # pad
    
    
def mytorchtext(texts_split, vocab, sent_num_steps, token_num_steps):
    """Convert the data into a vocabulary index and output the number of sentences and words in the text"""
    texts_features = []
    textsSent_valid_len = []
    textsToken_valid_len = []
    for text in vocab[texts_split]:
        sents_features = []
        sents_valid_len = []
        for sent in text:
            tokens_features, sent_valid_len = truncate_pad(sent, token_num_steps, vocab['<pad>'])
            sents_features.append(tokens_features)
            sents_valid_len.append(sent_valid_len)
        # If the number of sentences is less than the limit, add 0 at the end
        if len(sents_valid_len) < sent_num_steps:
            sents_valid_len += [0] * (sent_num_steps - len(sents_valid_len))
        else:
            sents_valid_len = sents_valid_len[:sent_num_steps]
        sents_features,  text_valid = sent_truncate_pad(sents_features, sent_num_steps, token_num_steps, vocab['<pad>'])
        texts_features.append(sents_features)
        textsToken_valid_len.append(sents_valid_len)
        textsSent_valid_len.append(text_valid)
        
    return texts_features, textsSent_valid_len, textsToken_valid_len
    
    
def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
    
    
def load_data(train_data, test_data, text_col_name, label_col_name, batch_size, 
              sent_num_steps, token_num_steps, dev_data=None, min_freq=None):
    train_texts_split = []
    for text in split_sentences(train_data[text_col_name], ' <_sentence_separator_> '):
        train_texts_split.append(tokenize(text, token='word'))
        
    test_texts_split = []
    for text in split_sentences(test_data[text_col_name], ' <_sentence_separator_> '):
        test_texts_split.append(tokenize(text, token='word'))
    
    vocab = Vocab(train_texts_split, min_freq, reserved_tokens=['<pad>'])
    
    train_texts_features, train_textsSent_valid_len, train_textsToken_valid_len = mytorchtext(train_texts_split, vocab, sent_num_steps, token_num_steps)
    
    test_texts_features, test_textsSent_valid_len, test_textsToken_valid_len = mytorchtext(test_texts_split, vocab, sent_num_steps, token_num_steps)
        
                                
    train_iter = load_array((torch.tensor(train_texts_features), 
                             torch.tensor(train_textsSent_valid_len), 
                             torch.tensor(train_textsToken_valid_len),
                             torch.tensor(train_data[label_col_name].to_numpy())), batch_size)
    
    test_iter = load_array((torch.tensor(test_texts_features), 
                            torch.tensor(test_textsSent_valid_len), 
                            torch.tensor(test_textsToken_valid_len),
                            torch.tensor(test_data[label_col_name].to_numpy())), batch_size, is_train=False)
                            
    if dev_data is not None:
        dev_texts_split = []
        for text in split_sentences(dev_data[text_col_name], ' <_sentence_separator_> '):
            dev_texts_split.append(tokenize(text, token='word'))
            
        dev_texts_features, dev_textsSent_valid_len, dev_textsToken_valid_len = mytorchtext(dev_texts_split, vocab, sent_num_steps, token_num_steps)
        
        dev_iter = load_array((torch.tensor(dev_texts_features), 
                               torch.tensor(dev_textsSent_valid_len), 
                               torch.tensor(dev_textsToken_valid_len),
                               torch.tensor(dev_data[label_col_name].to_numpy())), batch_size, is_train=False)
                               
    else:
        dev_iter = None
        
    return train_iter, dev_iter, test_iter, vocab