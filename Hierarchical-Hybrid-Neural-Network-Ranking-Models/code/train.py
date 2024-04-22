# coding=utf-8

import warnings
warnings.filterwarnings('ignore')

from operator import itemgetter
from collections import Counter
from tqdm import tqdm
import time

import torch
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    cohen_kappa_score,
)
from utils import save, get_pairs, text_to_pairs

## Define a function to validate a dataset once
def evaluate(args, model, iterator, criterion):
    epoch_loss = 0; train_num = 0; adjacent_corrects = 0
    model.eval()
    progress = tqdm(range(len(iterator)))
    with torch.no_grad(): # Prohibit gradient calculation
        true_label = torch.tensor([]); pre_label = torch.tensor([])
        for batch_idx, batch_item in enumerate(iterator):
            words, nSents, nWords, label = batch_item
            words = words.to(args.device)
            nSents = nSents.to(args.device)
            nWords = nWords.to(args.device)
            y = label.to(args.device)
            pre = model.forward(words, nSents, nWords)
            loss = criterion(pre, y)
            pre_lab = torch.argmax(pre,1)
            true_label = torch.cat([true_label, y.to('cpu')])
            pre_label = torch.cat([pre_label, pre_lab.to('cpu')])
            for i in range(len(pre_lab)):
                if pre_lab[i] == y[i] or pre_lab[i] == y[i]+1 or pre_lab[i] == y[i]-1:
                    adjacent_corrects += 1
            train_num += len(y) ## Sample quantity
            epoch_loss += loss.item()
            
            progress.set_description(
                'Evaluaton: {:.3f} | Loss: {:.5f}'.format(
                    (batch_idx + 1) / len(iterator),
                    loss.detach().item()
                )
            )
            progress.update()
            
        # metrics
        accuracy = accuracy_score(true_label, pre_label)
        adj_acc = float(adjacent_corrects) / train_num
        weighted_f1 = f1_score(true_label, pre_label, average='weighted')
        precision = precision_score(true_label, pre_label, average='weighted')
        recall = recall_score(true_label, pre_label, average='weighted')
        qwk = cohen_kappa_score(true_label, pre_label, weights='quadratic')
        ## Average loss of all samples
        epoch_loss = epoch_loss / train_num
        
    model.train()
    return epoch_loss, accuracy, adj_acc, weighted_f1, precision, recall, qwk
    
    
## Define a function to validate a dataset once
def rank_evaluate(args, model, iterator, label_map, criterion, dataset):
    epoch_loss = 0; total_num = 0; adjacent_corrects = 0
    model.eval()
    progress = tqdm(range(len(iterator)))
    # Determine the ranking of each index
    rank_dict = {index : [] for index in range(len(dataset))}
    with torch.no_grad(): # Prohibit gradient calculation
        for batch_idx, batch_item in enumerate(iterator):
            words, nSents, nWords, label, index = batch_item
            words = words.to(args.device)
            nSents = nSents.to(args.device)
            nWords = nWords.to(args.device)
            indexs = index
            
            pairs = get_pairs(torch.arange(args.num_class))
            y = text_to_pairs(label, pairs)
            y_result = y[:,0] - y[:, 1]
            y_result = y_result.reshape(-1)
            # Use torch Tensor.apply_ mapping
            y_result.apply_(lambda x: label_map[x])
            y_result = y_result.to(args.device)
            
            pre = model.forward(words, nSents, nWords, pairs)
            loss = criterion(pre, y_result.long())
            epoch_loss += loss.item()
            pre_lab = torch.argmax(pre,1)
            
            indexs = text_to_pairs(indexs, pairs)[:,0]
            # Get each batch index
            batch_index = set(indexs.tolist())
            
            # Setup dictionary of index to 'score'
            dict_of_indexs = {index: 0 for index in batch_index}
            
            # Get relative rankings based off comparisons
            for j in range(len(pre_lab)):
                dict_of_indexs[indexs[j].item()] += pre_lab[j]
            
            # save rank of index in each batch
            curr_rank = 0
            for index, _ in sorted(dict_of_indexs.items(), key=itemgetter(1)):
                rank_dict[index].append(curr_rank)
                curr_rank += 1
                
            progress.set_description(
                'Evaluaton: {:.3f} | Loss: {:.5f}'.format(
                    (batch_idx + 1) / len(iterator),
                    loss.detach().item()
                )
            )
            progress.update()
            
            
        # Calculate the readability level by counting the tags that appear most frequently in each index in rank_dict
        predict = []
        for index in range(len(dataset)):
            counter = Counter(rank_dict[index])
            predict.append(counter.most_common(1)[0][0])
            
        for i in range(len(predict)):
            if predict[i] == dataset['labelcode'].values[i] or predict[i] == dataset['labelcode'].values[i]+1 \
            or predict[i] == dataset['labelcode'].values[i]-1:
                adjacent_corrects += 1
        total_num += len(predict) ## Sample quantity
        
        # metrics
        true_label = dataset['labelcode'].values
        pre_label = np.array(predict)
        accuracy = accuracy_score(true_label, pre_label)
        adj_acc = float(adjacent_corrects) / total_num
        weighted_f1 = f1_score(true_label, pre_label, average='weighted')
        precision = precision_score(true_label, pre_label, average='weighted')
        recall = recall_score(true_label, pre_label, average='weighted')
        qwk = cohen_kappa_score(true_label, pre_label, weights='quadratic')
        ## Average loss of all samples
        epoch_loss = epoch_loss / total_num
        
    model.train()
    return epoch_loss, accuracy, adj_acc, weighted_f1, precision, recall, qwk
    
    
## Define a function that trains a dataset for one round
def train_epoch(args, model, iterator, optimizer, criterion, train_progress, 
                logger, k_cnt, l_cnt, start_time, epoch):
    epoch_loss = 0; train_num = 0
    model.train()
    for batch_idx, batch_item in enumerate(iterator['train']):
        words, nSents, nWords, label = batch_item
        words = words.to(args.device)
        nSents = nSents.to(args.device)
        nWords = nWords.to(args.device)
        y = label.to(args.device)
        
        optimizer.zero_grad()
        pre = model.forward(words, nSents, nWords)
        loss = criterion(pre, y)
        pre_lab = torch.argmax(pre,1)
        train_num += len(y) ## Sample quantity
        loss.backward()
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
        # for every portion of epoch, evaluate on train/test set each
        if (batch_idx + 1) % (len(iterator['train']) // args.n_eval_per_epoch) == 0:
            if args.do_evaluate:
                train_loss, train_acc, train_adj_acc, train_f1, train_p, train_r, train_qwk = evaluate(
                    args, model, iterator['train'], criterion)
                test_loss, test_acc, test_adj_acc, test_f1, test_p, test_r, test_qwk = evaluate(
                    args, model, iterator['test'], criterion)
                print()
                print('train accuracy:', train_acc, 'train f1:', train_f1)
                print('test accuracy:', test_acc, 'test f1:', test_f1)
                logger.info('TRAIN SET    | Epoch: {:.3f} | Accuracy: {:.4f} | F1: {:.4f} | QWK: {:.4f}'.format(
                    epoch + ((batch_idx + 1) / len(iterator['train'])),
                    train_acc,
                    train_f1,
                    train_qwk
                ))
                logger.info('TESR SET | Epoch: {:.3f} | Accuracy: {:.4f} | F1: {:.4f} | QWK: {:.4f}'.format(
                    epoch + ((batch_idx + 1) / len(iterator['train'])),
                    test_acc,
                    test_f1,
                    test_qwk
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
                    test_acc,
                    test_adj_acc,
                    test_f1,
                    test_p,
                    test_r,
                    test_qwk,
                    time.time() - start_time
                ]
            else:
                info = ['no evaluation done']
            save(args, model, info, k_cnt, l_cnt)
            l_cnt += 1
    
    ## Average loss and accuracy of all samples
    epoch_loss = epoch_loss / train_num
    return epoch_loss, l_cnt


## Define a function that trains a dataset for one round
def rank_train_epoch(args, model, dataset, iterator, label_map, optimizer, criterion, train_progress, logger, k_cnt, l_cnt, start_time, epoch):
    epoch_loss = 0; train_num = 0
    #train_corrects = 0; epoch_acc = 0
    model.train()
    for batch_idx, batch_item in enumerate(iterator['train']):
        words, nSents, nWords, label, index = batch_item
        words = words.to(args.device)
        nSents = nSents.to(args.device)
        nWords = nWords.to(args.device)
        # context = context.to(device)
        # label = label.to(device)
        
        pairs = get_pairs(torch.arange(args.num_class))
        y = text_to_pairs(label, pairs)
        y_result = y[:,0] - y[:, 1]
        y_result = y_result.reshape(-1)
        # Use torch Tensor.apply_ mapping
        y_result.apply_(lambda x: label_map[x])
        y_result = y_result.to(args.device)
        
        optimizer.zero_grad()
        pre = model.forward(words, nSents, nWords, pairs)
        loss = criterion(pre, y_result.long())
        pre_lab = torch.argmax(pre,1)
        # train_corrects += torch.sum(pre_lab == y_result)
        train_num += len(y_result) ## Sample quantity
        loss.backward()
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
        # for every portion of epoch, evaluate on train/test set each
        if (batch_idx + 1) % (len(iterator['train']) // args.n_eval_per_epoch) == 0:
            if args.do_evaluate:
                train_loss, train_acc, train_adj_acc, train_f1, train_p, train_r, train_qwk = rank_evaluate(
                    args, model, iterator['train'], label_map, criterion, dataset['train'])
                test_loss, test_acc, test_adj_acc, test_f1, test_p, test_r, test_qwk = rank_evaluate(
                    args, model, iterator['test'], label_map, criterion, dataset['test'])
                print()
                print('train accuracy:', train_acc, 'train f1:', train_f1)
                print('test accuracy:', test_acc, 'test f1:', test_f1)
                logger.info('TRAIN SET    | Epoch: {:.3f} | Accuracy: {:.4f} | F1: {:.4f} | QWK: {:.4f}'.format(
                    epoch + ((batch_idx + 1) / len(iterator['train'])),
                    train_acc,
                    train_f1,
                    train_qwk
                ))
                logger.info('TESR SET | Epoch: {:.3f} | Accuracy: {:.4f} | F1: {:.4f} | QWK: {:.4f}'.format(
                    epoch + ((batch_idx + 1) / len(iterator['train'])),
                    test_acc,
                    test_f1,
                    test_qwk
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
                    test_acc,
                    test_adj_acc,
                    test_f1,
                    test_p,
                    test_r,
                    test_qwk,
                    time.time() - start_time
                ]
            else:
                info = ['no evaluation done']
            save(args, model, info, k_cnt, l_cnt)
            l_cnt += 1
    
    ## 所有样本的平均损失和精度
    epoch_loss = epoch_loss / train_num
    # epoch_acc = train_corrects.double().item() / train_num
    return epoch_loss, l_cnt
    
    
class Trainer(object):
    def __init__(self, args, model, optimizer, data_iter, criterion, k_cnt):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.data_iter = data_iter
        self.criterion = criterion
        self.num_train_optimization_steps = len(self.data_iter['train']) * args.epochs
        self.k_cnt = k_cnt

        # train
        self.train_progress = tqdm(range(self.num_train_optimization_steps))

    def train(self, logger):
        print("Number of train examples: ", len(self.data_iter['train'].dataset))
        print("Batch size:", self.data_iter['train'].batch_size)
        print("Num of steps:", self.num_train_optimization_steps)

        l_cnt = 0
        for epoch in range(self.args.epochs):
            start_time = time.time()
            train_avg_loss, l_cnt = train_epoch(self.args, self.model, self.data_iter, self.optimizer, self.criterion, 
                                                self.train_progress, logger, self.k_cnt, l_cnt, start_time, epoch)
            if self.args.n_eval_per_epoch != 1:
                test_loss, test_acc, test_adj_acc, test_f1, test_p, test_r, test_qwk = evaluate(
                    self.args, self.model, self.data_iter['test'], self.criterion)
                info = [epoch+1,test_loss, test_acc, test_adj_acc, test_f1, test_p, test_r, test_qwk, time.time() - start_time]
                save(self.args, self.model, info, self.k_cnt, l_cnt)
                l_cnt += 1


class Rank_Trainer(object):
    def __init__(self, args, model, optimizer, dataset, data_iter, label_map, criterion, k_cnt):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.data_iter = data_iter
        self.label_map =  label_map
        self.criterion = criterion
        self.num_train_optimization_steps = len(self.data_iter['train']) * args.epochs
        self.k_cnt = k_cnt

        # train
        self.train_progress = tqdm(range(self.num_train_optimization_steps))

    def train(self, logger):
        print("Number of train examples: ", len(self.data_iter['train'].dataset))
        print("Batch size:", self.data_iter['train'].batch_size)
        print("Num of steps:", self.num_train_optimization_steps)

        l_cnt = 0
        for epoch in range(self.args.epochs):
            start_time = time.time()
            train_avg_loss, l_cnt = rank_train_epoch(self.args, self.model, self.dataset, self.data_iter, self.label_map, 
                                              self.optimizer, self.criterion, self.train_progress, logger, self.k_cnt, l_cnt, start_time, epoch)
            if self.args.n_eval_per_epoch != 1:
                test_loss, test_acc, test_adj_acc, test_f1, test_p, test_r, test_qwk = rank_evaluate(
                    self.args, self.model, self.data_iter['test'], self.label_map, self.criterion, self.dataset['test'])
                info = [epoch+1,test_loss, test_acc, test_adj_acc, test_f1, test_p, test_r, test_qwk, time.time() - start_time]
                save(self.args, self.model, info, self.k_cnt, l_cnt)
                l_cnt += 1