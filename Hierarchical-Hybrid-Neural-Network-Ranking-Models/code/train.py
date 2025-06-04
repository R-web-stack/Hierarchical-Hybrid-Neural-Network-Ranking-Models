# coding=utf-8

import warnings
warnings.filterwarnings('ignore')

from operator import itemgetter
from collections import Counter
from tqdm import tqdm
import time

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    cohen_kappa_score,
)
from utils import get_pairs, text_to_pairs, save
from model_layer import sequence_mask

    
# TSA
def get_tsa_thresh(args, schedule, global_step, num_train_steps, start, end):
    training_progress = torch.tensor(float(global_step) / float(num_train_steps))
    if schedule == 'linear_schedule':
        threshold = training_progress
    elif schedule == 'exp_schedule':
        scale = 5
        threshold = torch.exp((training_progress - 1) * scale)
    elif schedule == 'log_schedule':
        scale = 5
        threshold = 1 - torch.exp((-training_progress) * scale)
    output = threshold * (end - start) + start
    return output.to(args.device)


## 定义一个对数据集验证一轮的函数
def evaluate(args, model, iterator, criterion, epoch):
    epoch_loss = 0; train_num = 0; adjacent_corrects = 0
    model.eval()
    progress = tqdm(range(len(iterator)))
    # 记录每个文本的句子等级
    sentences_level_dict = {'index': [], 'sentences_level': []}
    with torch.no_grad(): # 禁止梯度计算
        true_label = torch.tensor([]); pre_label = torch.tensor([])
        for batch_idx, batch_item in enumerate(iterator):
            words, nSents, nWords, label, index = batch_item
            words = words.to(args.device)
            nSents = nSents.to(args.device)
            nWords = nWords.to(args.device)
            indexs = index
            y = label.to(args.device)
            
            pre_logit, sentence_weights, sentence_vectors = model.forward(words, nSents, nWords)
            loss = criterion(pre_logit, y)
            loss = torch.mean(loss)
            pre_lab = torch.argmax(pre_logit,1)
            true_label = torch.cat([true_label, y.to('cpu')])
            pre_label = torch.cat([pre_label, pre_lab.to('cpu')])
            for i in range(len(pre_lab)):
                if pre_lab[i] == y[i] or pre_lab[i] == y[i]+1 or pre_lab[i] == y[i]-1:
                    adjacent_corrects += 1
            train_num += len(y) ## 样本数量
            epoch_loss += loss.item()
            
            # 预测并记录句子可读性等级
            sentence_pre = model.difficulty_matrix(sentence_vectors)
            pre_sent_lab = torch.argmax(sentence_pre, -1).cpu()
            nSents_cpu = nSents.cpu()
            pre_sent_lab = [text[:nSents_cpu[index]].tolist() for index, text in enumerate(pre_sent_lab)]
            sentences_level_dict['index'] = sentences_level_dict['index'] + indexs.tolist()
            sentences_level_dict['sentences_level'] = sentences_level_dict['sentences_level'] + pre_sent_lab
            
            progress.set_description(
                'Evaluaton: {:.3f} | Loss: {:.5f}'.format(
                    (batch_idx + 1) / len(iterator),
                    loss.detach().item()
                )
            )
            progress.update()
            
        # 统计各个文本句子的可读性等级
        sentences_df = pd.DataFrame(sentences_level_dict)
        save_path = args.save_sentence_dir + f'_{epoch}.csv'
        sentences_df.to_csv(save_path, index=False)
        
        # metrics
        accuracy = accuracy_score(true_label, pre_label)
        adj_acc = float(adjacent_corrects) / train_num
        weighted_f1 = f1_score(true_label, pre_label, average='weighted')
        precision = precision_score(true_label, pre_label, average='weighted')
        recall = recall_score(true_label, pre_label, average='weighted')
        qwk = cohen_kappa_score(true_label, pre_label, weights='quadratic')
        ## 所有样本的平均损失
        epoch_loss = epoch_loss / train_num
        
    model.train()
    return epoch_loss, accuracy, adj_acc, weighted_f1, precision, recall, qwk
    
    
## 定义一个对数据集训练一轮的函数
def train_epoch(args, model, iterator, optimizer, criterion, sent_criterion, train_progress, 
                logger, k_cnt, l_cnt, start_time, epoch, total_steps, step):
    epoch_loss = 0; train_num = 0
    global_step = step
    model.train()
    for batch_idx, batch_item in enumerate(iterator['train']):
        words, nSents, nWords, label, index = batch_item
        words = words.to(args.device)
        nSents = nSents.to(args.device)
        nWords = nWords.to(args.device)
        y = label.to(args.device)
        
        optimizer.zero_grad()
        pre_logit, sentence_weights, sentence_vectors = model.forward(words, nSents, nWords)
        loss = criterion(pre_logit, y)
        # pre_lab = torch.argmax(pre,1)
        
        if args.tsa:
            tsa_thresh = get_tsa_thresh(args, args.tsa, global_step, total_steps, start=1./args.num_class, end=1)
            larger_than_threshold = torch.exp(-loss) > tsa_thresh   # prob = exp(log_prob), prob > tsa_threshold
            loss_mask = torch.ones_like(y, dtype=torch.float32) * (1 - larger_than_threshold.type(torch.float32))
            loss = torch.sum(loss * loss_mask, dim=-1) / torch.max(torch.sum(loss_mask, dim=-1), torch.tensor(1.).to(args.device))
        else:
            loss = torch.mean(loss)
            
        # 模型预测的文本等级概率
        pre_prob = F.softmax(pre_logit, dim=-1)
        # confidence-based masking
        if args.uda_confidence_thresh != -1:
            unsup_loss_mask = torch.max(pre_prob, dim=-1)[0] > args.uda_confidence_thresh
            unsup_loss_mask = unsup_loss_mask.type(torch.float32)
        else:
            unsup_loss_mask = torch.ones(len(pre_prob), dtype=torch.float32)
        unsup_loss_mask = unsup_loss_mask.to(args.device)
        # 取句子注意力权重的平均值
        aggregated_weights = torch.mean(sentence_weights, dim=-1)
        weights = torch.nn.functional.softmax(sequence_mask(aggregated_weights, nSents, value=-1e6), dim=-1).unsqueeze(-1)
        # 对句子预测
        sentence_pre = model.difficulty_matrix(sentence_vectors)
        # 聚合成文本等级概率
        text_pre = (sentence_pre * weights).sum(dim=1)
        # softmax temperature controlling
        uda_softmax_temp = args.uda_softmax_temp if args.uda_softmax_temp > 0 else 1.
        text_prob = F.log_softmax(text_pre / uda_softmax_temp, dim=-1)
        # 计算句子聚合的文本等级概率和模型预测的文本等级概率的损失
        unsup_loss = torch.sum(sent_criterion(text_prob, pre_prob.detach()), dim=-1)
        unsup_loss = torch.sum(unsup_loss * unsup_loss_mask, dim=-1) / torch.max(torch.sum(unsup_loss_mask, dim=-1), torch.tensor(1.).to(args.device))
        
        # 最终损失：文本 + 句子
        final_loss = loss + unsup_loss
        
        global_step += 1
        train_num += len(y) ## 样本数量
        final_loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        train_progress.set_description(
            'Epoch: {:.3f} | Loss: {:.5f}'.format(
                epoch + ((batch_idx + 1) / len(iterator['train'])),
                loss.detach().item()
            )
        )
        train_progress.update()
        
        # evaluate
        # for every portion of epoch, evaluate on train/validation/test set each
        if (batch_idx + 1) % (len(iterator['train']) // args.n_eval_per_epoch) == 0:
            if args.do_evaluate:
                train_loss, train_acc, train_adj_acc, train_f1, train_p, train_r, train_qwk = evaluate(
                    args, model, iterator['train'], criterion, epoch)
                print()
                print('train accuracy:', train_acc, 'train f1:', train_f1)
                logger.info('TRAIN SET    | Epoch: {:.3f} | Loss: {:.5f} | Accuracy: {:.4f} | Adj_Acc: {:.4f} | F1: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | QWK: {:.4f}'.format(
                    epoch + ((batch_idx + 1) / len(iterator['train'])),
                    train_loss, 
                    train_acc,
                    train_adj_acc,
                    train_f1,
                    train_p,
                    train_r,
                    train_qwk
                ))
                info = [
                        epoch + ((batch_idx + 1) / len(iterator['train'])),
                        train_loss,
                        train_acc,
                        train_adj_acc,
                        train_f1,
                        train_p,
                        train_r,
                        train_qwk,
                        time.time() - start_time
                    ]
            else:
                info = ['no evaluation done']
            # save(args, model, info, k_cnt, l_cnt)
            l_cnt += 1
    
    ## 所有样本的平均损失和精度
    epoch_loss = epoch_loss / train_num
    return epoch_loss, l_cnt, global_step
    
    
class Trainer(object):
    def __init__(self, args, model, optimizer, data_iter, criterion, sent_criterion, k_cnt):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.data_iter = data_iter
        self.criterion = criterion
        self.sent_criterion = sent_criterion
        self.num_train_optimization_steps = len(self.data_iter['train']) * args.epochs
        self.k_cnt = k_cnt
        self.global_step = 0

        # train
        self.train_progress = tqdm(range(self.num_train_optimization_steps))

    def train(self, logger):
        print("Number of train examples: ", len(self.data_iter['train'].dataset))
        print("Batch size:", self.data_iter['train'].batch_size)
        print("Num of steps:", self.num_train_optimization_steps)

        l_cnt = 0
        for epoch in range(self.args.epochs):
            start_time = time.time()
            train_avg_loss, l_cnt, global_step = train_epoch(self.args, self.model, self.data_iter, self.optimizer, self.criterion, self.sent_criterion, 
                                                self.train_progress, logger, self.k_cnt, l_cnt, start_time, epoch,
                                                self.num_train_optimization_steps, self.global_step)
            self.global_step = global_step
            if self.args.n_eval_per_epoch != 1:
                test_loss, test_acc, test_adj_acc, test_f1, test_p, test_r, test_qwk = evaluate(
                    self.args, self.model, self.data_iter['test'], self.criterion)
                info = [epoch+1,test_loss, test_acc, test_adj_acc, test_f1, test_p, test_r, test_qwk, time.time() - start_time]
                # save(self.args, self.model, info, self.k_cnt, l_cnt)
                l_cnt += 1