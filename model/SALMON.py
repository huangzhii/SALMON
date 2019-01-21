#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Zhi Huang
"""
import argparse, random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
import tables
import csv
import numpy as np
import json
from tqdm import tqdm
import gc
import copy

len_of_RNAseq = 57
len_of_miRNAseq = 12
len_of_cnv = 1
len_of_tmb = 1
len_of_clinical = 3

class SALMON(nn.Module):
    def __init__(self, input_dim, dropout_rate, label_dim):
        super(SALMON, self).__init__()
        hidden1 = 8
        hidden2 = 4
        
        if input_dim == len_of_RNAseq: # mRNAseq
            self.encoder1 = nn.Sequential(nn.Linear(input_dim, hidden1),nn.Sigmoid())
            self.classifier = nn.Sequential(nn.Linear(hidden1, label_dim),nn.Sigmoid())
            
        if input_dim == len_of_miRNAseq: # miRNAseq
            self.encoder2 = nn.Sequential(nn.Linear(input_dim, hidden2),nn.Sigmoid())
            self.classifier = nn.Sequential(nn.Linear(hidden2, label_dim),nn.Sigmoid())
            
        if input_dim == len_of_RNAseq + len_of_miRNAseq: # mRNAseq + miRNAseq
            self.encoder1 = nn.Sequential(nn.Linear(len_of_RNAseq, hidden1),nn.Sigmoid())
            self.encoder2 = nn.Sequential(nn.Linear(len_of_miRNAseq, hidden2),nn.Sigmoid())
            self.classifier = nn.Sequential(nn.Linear(hidden1 + hidden2, label_dim),nn.Sigmoid())
            
        if input_dim == len_of_RNAseq + len_of_miRNAseq + len_of_cnv + len_of_tmb: # mRNAseq + miRNAseq + CNB + TMB
            hidden_cnv, hidden_tmb = len_of_cnv, len_of_tmb
            self.encoder1 = nn.Sequential(nn.Linear(len_of_RNAseq, hidden1),nn.Sigmoid())
            self.encoder2 = nn.Sequential(nn.Linear(len_of_miRNAseq, hidden2),nn.Sigmoid())
            self.classifier = nn.Sequential(nn.Linear(hidden1 + hidden2 + hidden_cnv + hidden_tmb, label_dim),nn.Sigmoid())
                        
        if input_dim == len_of_RNAseq + len_of_miRNAseq + len_of_cnv + len_of_tmb + len_of_clinical: # mRNAseq + miRNAseq + CNB + TMB + clinical
            hidden_cnv, hidden_tmb, hidden_clinical = len_of_cnv, len_of_tmb, len_of_clinical
            self.encoder1 = nn.Sequential(nn.Linear(len_of_RNAseq, hidden1),nn.Sigmoid())
            self.encoder2 = nn.Sequential(nn.Linear(len_of_miRNAseq, hidden2),nn.Sigmoid())
            self.classifier = nn.Sequential(nn.Linear(hidden1 + hidden2 + \
                                            hidden_cnv + hidden_tmb + hidden_clinical, label_dim),nn.Sigmoid())
            
        if input_dim == len_of_cnv + len_of_tmb + len_of_clinical: # CNB + TMB + clinical
            hidden_cnv, hidden_tmb, hidden_clinical = len_of_cnv, len_of_tmb, len_of_clinical
            self.classifier = nn.Sequential(nn.Linear(hidden_cnv + hidden_tmb + hidden_clinical, label_dim),nn.Sigmoid())
        
        if input_dim == len_of_RNAseq + len_of_miRNAseq + len_of_clinical: # mRNAseq + miRNAseq + clinical
            hidden_clinical = len_of_clinical
            self.encoder1 = nn.Sequential(nn.Linear(len_of_RNAseq, hidden1),nn.Sigmoid())
            self.encoder2 = nn.Sequential(nn.Linear(len_of_miRNAseq, hidden2),nn.Sigmoid())
            self.classifier = nn.Sequential(nn.Linear(hidden1 + hidden2 + \
                                            hidden_clinical, label_dim),nn.Sigmoid())
        
    def forward(self, x):
        input_dim = x.shape[1]
        x_d = None
        if input_dim == len_of_RNAseq: # mRNAseq
            code1 = self.encoder1(x)
            lbl_pred = self.classifier(code1) # predicted label
            code = code1
            
        if input_dim == len_of_miRNAseq: # miRNAseq
            code2 = self.encoder2(x)
            lbl_pred = self.classifier(code2) # predicted label
            code = code2
            
        if input_dim == len_of_RNAseq + len_of_miRNAseq: # mRNAseq + miRNAseq
            code1 = self.encoder1(x[:,0:len_of_RNAseq])
            code2 = self.encoder2(x[:,len_of_RNAseq:])
            lbl_pred = self.classifier(torch.cat((code1, code2), 1)) # predicted label
            code = torch.cat((code1, code2), 1)
            
        if input_dim == len_of_RNAseq + len_of_miRNAseq + len_of_cnv + len_of_tmb: # mRNAseq + miRNAseq + CNB + TMB
            code1 = self.encoder1(x[:,0:len_of_RNAseq])
            code2 = self.encoder2(x[:,len_of_RNAseq: (len_of_RNAseq + len_of_miRNAseq)])
            lbl_pred = self.classifier(torch.cat((code1, code2, x[:,(len_of_RNAseq + len_of_miRNAseq):]), 1)) # predicted label
            code = torch.cat((code1, code2), 1)
                        
        if input_dim == len_of_RNAseq + len_of_miRNAseq + len_of_cnv + len_of_tmb + len_of_clinical: # mRNAseq + miRNAseq + CNB + TMB + clinical
            code1 = self.encoder1(x[:,0:len_of_RNAseq])
            code2 = self.encoder2(x[:,len_of_RNAseq: (len_of_RNAseq + len_of_miRNAseq)])
            lbl_pred = self.classifier(torch.cat((code1, code2, x[:, (len_of_RNAseq + len_of_miRNAseq):]), 1)) # predicted label
            code = torch.cat((code1, code2), 1)
            
        if input_dim == len_of_cnv + len_of_tmb + len_of_clinical: # CNB + TMB + clinical
            lbl_pred = self.classifier(x) # predicted label
            code = torch.FloatTensor([0])
            
        if input_dim == len_of_RNAseq + len_of_miRNAseq + len_of_clinical: # mRNAseq + miRNAseq + clinical
            code1 = self.encoder1(x[:,0:len_of_RNAseq])
            code2 = self.encoder2(x[:,len_of_RNAseq: (len_of_RNAseq + len_of_miRNAseq)])
            lbl_pred = self.classifier(torch.cat((code1, code2, x[:, (len_of_RNAseq + len_of_miRNAseq):]), 1)) # predicted label
            code = torch.cat((code1, code2), 1)
            
        return x_d, code, lbl_pred


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def accuracy_cox(hazards, labels):
    # This accuracy is based on estimated survival events against true survival events
    hazardsdata = hazards.cpu().numpy().reshape(-1)
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    labels = labels.data.cpu().numpy()
    correct = np.sum(hazards_dichotomize == labels)
    return correct / len(labels)

def cox_log_rank(hazards, labels, survtime_all):
    hazardsdata = hazards.cpu().numpy().reshape(-1)
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    survtime_all = survtime_all.data.cpu().numpy().reshape(-1)
    idx = hazards_dichotomize == 0
    labels = labels.data.cpu().numpy()
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return(pvalue_pred)
    
def CIndex(hazards, labels, survtime_all):
    labels = labels.data.cpu().numpy()
    concord = 0.
    total = 0.
    N_test = labels.shape[0]
    labels = np.asarray(labels, dtype=bool)
    for i in range(N_test):
        if labels[i] == 1:
            for j in range(N_test):
                if survtime_all[j] > survtime_all[i]:
                    total = total + 1
                    if hazards[j] < hazards[i]: concord = concord + 1
                    elif hazards[j] < hazards[i]: concord = concord + 0.5

    return(concord/total)
    
def CIndex_lifeline(hazards, labels, survtime_all):
    labels = labels.data.cpu().numpy()
    hazards = hazards.cpu().numpy().reshape(-1)
    return(concordance_index(survtime_all, -hazards, labels))
        
def frobenius_norm_loss(a, b):
    loss = torch.sqrt(torch.sum(torch.abs(a-b)**2))
    return loss

def test(model, datasets, whichset, batch_size, cuda, verbose):
    x = datasets[whichset]['x']
    e = datasets[whichset]['e']
    t = datasets[whichset]['t']
    X = torch.FloatTensor(x)
    OS_event = torch.LongTensor(e)
    OS = torch.FloatTensor(t)
    dataloader = DataLoader(X, batch_size=batch_size, num_workers=1, pin_memory=True, shuffle=False)
    lblloader = DataLoader(OS_event, batch_size=batch_size, num_workers=1, pin_memory=True, shuffle=False)
    OSloader = DataLoader(OS, batch_size=batch_size, num_workers=1, pin_memory=True, shuffle=False)
    lbl_pred_all = None
    lbl_all = None
    survtime_all = None
    code_final = None
    loss_nn_sum = 0
    model.eval()
    iter = 0
    for data, lbl, survtime in zip(dataloader, lblloader, OSloader):
        graph = data
        graph = Variable(graph)
        lbl = Variable(lbl)
        if cuda:
            model = model.cuda()
            graph = graph.cuda()
            lbl = lbl.cuda()
        # ===================forward=====================
        output, code, lbl_pred = model(graph)
        if iter == 0:
            lbl_pred_all = lbl_pred
            lbl_all = lbl
            survtime_all = survtime
            code_final = code
        else:
            lbl_pred_all = torch.cat([lbl_pred_all, lbl_pred])
            lbl_all = torch.cat([lbl_all, lbl])
            survtime_all = torch.cat([survtime_all, survtime])
            code_final = torch.cat([code_final, code])
            
        current_batch_len = len(survtime)
        R_matrix_test = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_matrix_test[i,j] = survtime[j] >= survtime[i]
    
        test_R = torch.FloatTensor(R_matrix_test)
        test_R = Variable(test_R)
        if cuda:
            test_R = test_R.cuda()
        test_ystatus = lbl
        theta = lbl_pred.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_nn = -torch.mean( (theta - torch.log(torch.sum( exp_theta*test_R ,dim=1))) * test_ystatus.float() )
        loss_nn_sum = loss_nn_sum + loss_nn.data.item()
        iter += 1
    code_final_4_original_data = code_final.data.cpu().numpy()
    acc_test = accuracy_cox(lbl_pred_all.data, lbl_all)
    pvalue_pred = cox_log_rank(lbl_pred_all.data, lbl_all, survtime_all)
    c_index = CIndex_lifeline(lbl_pred_all.data, lbl_all, survtime_all)
    if verbose > 0:
        print('\n[{:s}]\t\tloss (nn):{:.4f}'.format(whichset, loss_nn_sum),
                      'c_index: {:.4f}, p-value: {:.3e}'.format(c_index, pvalue_pred))
    return(code_final_4_original_data, loss_nn_sum, acc_test, \
           pvalue_pred, c_index, lbl_pred_all.data.cpu().numpy().reshape(-1), OS_event, survtime_all)
    
def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0, 0.5)
    
def train(datasets, num_epochs, batch_size, learning_rate, dropout_rate,
                        lambda_1, cuda, measure, verbose):
    

    x = datasets['train']['x']
    e = datasets['train']['e']
    t = datasets['train']['t']
    nodes_in = x.shape[1]
    
    X = torch.FloatTensor(x)
    OS_event = torch.LongTensor(e)
    OS = torch.FloatTensor(t)
        
    
    dataloader = DataLoader(X, batch_size=batch_size, num_workers=1, pin_memory=True, shuffle=False)
    lblloader = DataLoader(OS_event, batch_size=batch_size, num_workers=1, pin_memory=True, shuffle=False)
    OSloader = DataLoader(OS, batch_size=batch_size, num_workers=1, pin_memory=True, shuffle=False)
    
    
    
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(666)
    torch.manual_seed(666)
    random.seed(666)
    
    model = SALMON(nodes_in, dropout_rate, label_dim = 1)
        
    if cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    
    c_index_list = {}
    c_index_list['train'] = []
    c_index_list['test'] = []
    loss_nn_all = []
    pvalue_all = []
    c_index_all = []
    acc_train_all = []
    c_index_best = 0
    code_output = None
    

    for epoch in tqdm(range(num_epochs)):
        model.train()
        lbl_pred_all = None
        lbl_all = None
        survtime_all = None
        code_final = None
        loss_nn_sum = 0
        iter = 0
        gc.collect()
        for data, lbl, survtime in zip(dataloader, lblloader, OSloader):
            optimizer.zero_grad() # zero the gradient buffer
            graph = data
            if cuda:
                model = model.cuda()
                graph = graph.cuda()
                lbl = lbl.cuda()
            # ===================forward=====================
            output, code, lbl_pred = model(graph)
            
            if iter == 0:
                lbl_pred_all = lbl_pred
                survtime_all = survtime
                lbl_all = lbl
                code_final = code
            else:
                lbl_pred_all = torch.cat([lbl_pred_all, lbl_pred])
                lbl_all = torch.cat([lbl_all, lbl])
                survtime_all = torch.cat([survtime_all, survtime])
                code_final = torch.cat([code_final, code])
            # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
            # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
            current_batch_len = len(survtime)
            R_matrix_train = np.zeros([current_batch_len, current_batch_len], dtype=int)
            for i in range(current_batch_len):
                for j in range(current_batch_len):
                    R_matrix_train[i,j] = survtime[j] >= survtime[i]
        
            train_R = torch.FloatTensor(R_matrix_train)
            if cuda:
                train_R = train_R.cuda()
            train_ystatus = lbl
            
            theta = lbl_pred.reshape(-1)
            exp_theta = torch.exp(theta)
            
            loss_nn = -torch.mean( (theta - torch.log(torch.sum( exp_theta*train_R ,dim=1))) * train_ystatus.float() )

            l1_reg = None
            for W in model.parameters():
                if l1_reg is None:
                    l1_reg = torch.abs(W).sum()
                else:
                    l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)
            
            loss = loss_nn + lambda_1 * l1_reg
            if verbose > 0:
                print("\nloss_nn: %.4f, L1: %.4f" % (loss_nn, lambda_1 * l1_reg))
            loss_nn_sum = loss_nn_sum + loss_nn.data.item()
            # ===================backward====================
            loss.backward()
            optimizer.step()
            
            iter += 1
            torch.cuda.empty_cache()
        code_final_4_original_data = code_final.data.cpu().numpy()
        
        if measure or epoch == (num_epochs - 1):
            acc_train = accuracy_cox(lbl_pred_all.data, lbl_all)
            pvalue_pred = cox_log_rank(lbl_pred_all.data, lbl_all, survtime_all)
            c_index = CIndex_lifeline(lbl_pred_all.data, lbl_all, survtime_all)
            
            c_index_list['train'].append(c_index)
            if c_index > c_index_best:
                c_index_best = c_index
                code_output = code_final_4_original_data
            if verbose > 0:
                print('\n[Training]\t loss (nn):{:.4f}'.format(loss_nn_sum),
                      'c_index: {:.4f}, p-value: {:.3e}'.format(c_index, pvalue_pred))
            pvalue_all.append(pvalue_pred)
            c_index_all.append(c_index)
            loss_nn_all.append(loss_nn_sum)
            acc_train_all.append(acc_train)
            whichset = 'test'
            code_validation, loss_nn_sum, acc_test, pvalue_pred, c_index_pred, lbl_pred_all, OS_event, OS = \
                test(model, datasets, whichset, batch_size, cuda, verbose)
                
            c_index_list['test'].append(c_index_pred)
    return(model, loss_nn_all, pvalue_all, c_index_all, c_index_list, acc_train_all, code_output)
