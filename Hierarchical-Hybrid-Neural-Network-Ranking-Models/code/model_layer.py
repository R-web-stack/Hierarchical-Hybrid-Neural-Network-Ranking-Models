# coding=utf-8

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

class Bi_Rnn_Layer(nn.Module):
    def __init__(self, is_GRU, embedding_dim, hidden_dim, layer_dim, bidirectional, batch_first=True):
        """
        rnn:  LSTM or GRU
        embedding_dim: The dimension of word vectors
        hidden_dim: Number of RNN neurons
        layer_dim: The number of layers of RNN
        bidirectional: set bidirectional?
        batch_first: Is the first dimension of the input data batch
        """
        super().__init__()
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        if is_GRU:
            # GRU
            self.rnn = nn.GRU(embedding_dim, hidden_dim, layer_dim, bidirectional = self.bidirectional,
                            batch_first = self.batch_first)
        else:
            # LSTM
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, layer_dim, bidirectional = self.bidirectional,
                            batch_first = self.batch_first)
            
    def forward(self, word_embeds, nWords, begin_state=None):
        allSentsFlat = word_embeds.view(-1,word_embeds.size(2),word_embeds.size(3))
        nWordsFlat = nWords.view(-1)
        
        #Converting all 0 length sentences to length of 1
        #This is a hack to avoid the fact that pack_padded_sequence does not accept length 0 sequences
        #This hack should not be needed in the future as community of pytorch is working towards this feature:
        #https://github.com/pytorch/pytorch/issues/9681
        nWordsFlatNonZero= torch.Tensor.clone(nWordsFlat)
        nWordsFlatNonZero[nWordsFlatNonZero==0]=1
        
        order=torch.argsort(-nWordsFlat)
        orderInverse=torch.argsort(order)
        
        pack = nn.utils.rnn.pack_padded_sequence(allSentsFlat[order], nWordsFlatNonZero[order].cpu(), batch_first=True)
        output, hn = self.rnn(pack)
        hAllWordsFlat, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        hAllWordsFlat = hAllWordsFlat[orderInverse]
        
        #Converting all h of length 0 sentences to zeros, so that the hack (see above) does not affect the end result by any means
        hAllWordsFlat[nWordsFlat == 0] = 0 #the 0 gets broadcasted to fit the dimensions needed
        
        #Revert flattening back to original hierachy by splitting using number of sentences per text
        hAllWords = hAllWordsFlat.split(word_embeds.size(1), dim=0)
        hAllWords= torch.stack(hAllWords, dim=0) # this step is needed as split returns a tuple
        
        return hAllWords
        
        
def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences.

    Defined in :numref:`sec_seq2seq_decoder`"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X
    
    
def cnn_masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis.

    Defined in :numref:`sec_attention-scoring-functions`"""
    valid_lens = valid_lens.reshape(-1)
    # On the last axis, replace masked elements with a very large negative
    # value, whose exponentiation outputs 0
    X = sequence_mask(X, valid_lens, value=-1e6)
    return nn.functional.softmax(X, dim=-1)
    
    

class CNN_Attention_Layer(nn.Module):
    def __init__(self, embedding_dim, n_filters, filter_sizes, filter_padding):
        """
        n_filters: The number of convolutional kernels
        filter_sizes: The size of convolutional kernels
        filter_padding: The number of convolutional kernel padding
        """
        super().__init__()
        ## Convolution operation
        self.conv = nn.Conv2d(in_channels = 1, out_channels = n_filters, 
                  kernel_size = (filter_sizes, embedding_dim+2*filter_padding), padding = filter_padding)
        self.relu = nn.ReLU()
        
    def forward(self, word_embeds, nWords):
        allSentsFlat = word_embeds.view(-1,word_embeds.size(2),word_embeds.size(3))
        nWordsFlat = nWords.view(-1)
        #allSentsFlat = [batch size*sent_num, sent len, emb dim]
        embedded = allSentsFlat.unsqueeze(1)
        #embedded = [batch size*sent_num, 1, sent len, emb dim]
        conved = self.relu(self.conv(embedded)).squeeze(3)
        #conved_n = [batch size*sent_num, n_filters, sent len - filter_sizes + 2*filter_padding + 1]
        pooled_avg = torch.mean(conved, dim=1, keepdim=True).squeeze(1)
        #pooled_n = [batch size*sent_num, sent len - filter_sizes + 2*filter_padding + 1]
        weight = cnn_masked_softmax(pooled_avg, nWordsFlat)
        # weight = [batch size*sent_num, sent len - filter_sizes + 2*filter_padding + 1]
        attention_output = torch.bmm(weight.unsqueeze(1),allSentsFlat).reshape(word_embeds.size(0), word_embeds.size(1), -1)
        return attention_output
        
        
class Mdim_CNN_Context(nn.Module):
    def __init__(self, embedding_dim, n_filters, filter_sizes, filter_padding, n_h):
        """
        n_filters: The number of convolutional kernels
        filter_sizes: The size of convolutional kernels
        filter_padding: The number of convolutional kernel padding
        """
        super().__init__()
        ## Convolution operation
        self.conv = nn.Conv2d(in_channels = 1, out_channels = n_filters, 
                  kernel_size = (filter_sizes, embedding_dim+2*filter_padding), padding = filter_padding)
        self.MH_Attention = MultiHeadAttention(key_size=embedding_dim, query_size=embedding_dim, value_size=n_filters, 
                                               num_hiddens=embedding_dim, num_heads=n_h, dropout=0.3)
        # self.relu = nn.ReLU(inplace=False)
        
    def forward(self, word_embeds, nWords):
        allSentsFlat = word_embeds.view(-1,word_embeds.size(2),word_embeds.size(3))
        nWordsFlat = nWords.view(-1)
        #allSentsFlat = [batch size*sent_num, sent len, emb dim]
        
        embedded = allSentsFlat.unsqueeze(1)
        #embedded = [batch size*sent_num, 1, sent len, emb dim]
        conved = self.conv(embedded).squeeze(3)
        #conved_n = [batch size*sent_num, n_filters, sent len - filter_sizes + 2*filter_padding + 1]
        context = conved.permute(0,2,1)
        #context = [batch size*sent_num, sent len - filter_sizes + 2*filter_padding + 1, n_filters]
        context_weight = self.MH_Attention(allSentsFlat, allSentsFlat, context, nWordsFlat)
        # context_weight = torch.relu(context_weight)
        #context_weight = [batch size*sent_num, sent len - filter_sizes + 2*filter_padding + 1, emb dim]
        weight_mask = sequence_mask(context_weight, nWordsFlat, value=-1e6)
        weight = F.softmax(weight_mask, dim=-2)
        
        attention_output = weight * allSentsFlat
        attention_output = attention_output.reshape(word_embeds.shape)
        attention_output = torch.sum(attention_output, dim=-2).reshape(word_embeds.size(0), word_embeds.size(1), -1)
        
        return attention_output
        
    
    
def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis.

    Defined in :numref:`sec_attention-scoring-functions`"""
    # `X`: 3D tensor, `valid_lens`: 1D or 2D tensor
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)
        
        
class DotProductAttention(nn.Module):
    """Scaled dot product attention.

    Defined in :numref:`subsec_additive-attention`"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # Shape of `queries`: (`batch_size`, no. of queries, `d`)
    # Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
    # Shape of `values`: (`batch_size`, no. of key-value pairs, value
    # dimension)
    # Shape of `valid_lens`: (`batch_size`,) or (`batch_size`, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Set `transpose_b=True` to swap the last two dimensions of `keys`
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
        
        
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
        
        
class MultiHeadAttention(nn.Module):
    """Multi-head attention.

    Defined in :numref:`sec_multihead-attention`"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # Shape of `queries`, `keys`, or `values`:
        # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`)
        # Shape of `valid_lens`:
        # (`batch_size`,) or (`batch_size`, no. of queries)
        # After transposing, shape of output `queries`, `keys`, or `values`:
        # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
        # `num_hiddens` / `num_heads`)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for
            # `num_heads` times, then copy the next item, and so on
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # Shape of `output`: (`batch_size` * `num_heads`, no. of queries,
        # `num_hiddens` / `num_heads`)
        output = self.attention(queries, keys, values, valid_lens)

        # Shape of `output_concat`:
        # (`batch_size`, no. of queries, `num_hiddens`)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)
        
        
def transpose_qkv(X, num_heads):
    """Transposition for parallel computation of multiple attention heads.

    Defined in :numref:`sec_multihead-attention`"""
    # Shape of input `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`).
    # Shape of output `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_heads`,
    # `num_hiddens` / `num_heads`)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # Shape of output `X`:
    # (`batch_size`, `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    X = X.permute(0, 2, 1, 3)

    # Shape of `output`:
    # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """Reverse the operation of `transpose_qkv`.

    Defined in :numref:`sec_multihead-attention`"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)
    
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
        
        
#layer normalization [(cite)](https://arxiv.org/abs/1607.06450). do on
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        
        
class customizedModule(nn.Module):
    def __init__(self):
        super(customizedModule, self).__init__()

    # linear transformation (w/ initialization) + activation + dropout
    def customizedLinear(self, in_dim, out_dim, activation=None, dropout=False):
        cl = nn.Sequential(nn.Linear(in_dim, out_dim))
        nn.init.xavier_uniform(cl[0].weight)
        nn.init.constant(cl[0].bias, 0)

        if activation is not None:
            cl.add_module(str(len(cl)), activation)
        if dropout:
            cl.add_module(str(len(cl)), nn.Dropout(p=self.args.dropout))

        return cl
        
        
        
# That is, the output of each sub-layer is $\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$, where $\mathrm{Sublayer}(x)$ is the function implemented by the sub-layer itself.  We apply dropout [(cite)](http://jmlr.org/papers/v15/srivastava14a.html) to the output of each sub-layer, before it is added to the sub-layer input and normalized.
# To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension $d_{\text{model}}=512$.
class SublayerConnection(customizedModule):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout, input_dim):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.init_mBloSA(input_dim)

    def init_mBloSA(self, input_dim):
        self.g_W1 = self.customizedLinear(input_dim, input_dim)
        self.g_W2 = self.customizedLinear(input_dim, input_dim)
        self.g_b = nn.Parameter(torch.zeros(input_dim))

        self.g_W1[0].bias.requires_grad = False
        self.g_W2[0].bias.requires_grad = False

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        G = F.sigmoid(self.g_W1(x) + self.g_W2(self.dropout(sublayer(self.norm(x)))) + self.g_b)
        # (batch, m, word_dim)
        ret = G * x + (1 - G) * self.dropout(sublayer(self.norm(x)))
        return ret
        
        
# Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network.
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn      
        self.feed_forward = feed_forward    
        self.sublayer = clones(SublayerConnection(size, dropout, size), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        ret = self.sublayer[1](x, self.feed_forward)
        return ret
        
        
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    
    
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
        
        
        
class s2tSA(customizedModule):
    def __init__(self, hidden_size):
        super(s2tSA, self).__init__()

        self.s2t_W1 = self.customizedLinear(hidden_size, hidden_size, activation=nn.ReLU())
        self.s2t_W = self.customizedLinear(hidden_size, hidden_size)

    def forward(self, x):
        """
        source2token self-attention module
        :param x: (batch, (block_num), seq_len, hidden_size)
        :return: s: (batch, (block_num), hidden_size)
        """

        # (batch, (block_num), seq_len, word_dim)
        f = self.s2t_W1(x)
        f = F.softmax(self.s2t_W(f), dim=-2)
        # (batch, (block_num), word_dim)
        s = torch.sum(f * x, dim=-2)
        return s, f
        
        
        
        
# MDEM

def transpose(X, num_heads):
    """Transposition for parallel computation of multiple attention heads.

    Defined in :numref:`sec_multihead-attention`"""
    # Shape of input `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`).
    # Shape of output `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_heads`,
    # `num_hiddens` / `num_heads`)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # Shape of output `X`:
    # (`batch_size`, `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    X = X.permute(0, 2, 1, 3)

    # Shape of `output`:
    # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    return X
    
    
    
class MHDifficultyMatrix(nn.Module):
    def __init__(self, args, imput_dim, num_heads, **kwargs):
        super(MHDifficultyMatrix, self).__init__(**kwargs)
        self.C = nn.Parameter(torch.Tensor(1, args.num_class, num_heads, imput_dim//num_heads).permute(0, 2, 1, 3))
        init.xavier_uniform_(self.C)  # 初始化
        self.num_heads = num_heads
        
    def forward(self, X):
        X = transpose(X, self.num_heads)
        sentence_logit = torch.matmul(X, self.C.transpose(2,3))
        sentence_logit = torch.sum(sentence_logit, dim=1)
        return sentence_logit