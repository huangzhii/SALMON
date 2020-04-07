#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Zhi Huang
"""

import sys, os
sys.path.append("/home/zhihuan/Documents/SALMON/model")
import SALMON
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
from imblearn.over_sampling import RandomOverSampler
from lifelines.statistics import logrank_test
import json
import tables
import logging
import csv
import numpy as np
import optunity
import pickle
import time
from sklearn.model_selection import KFold
from sklearn import preprocessing
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs to train for. Default: 100")
    parser.add_argument('--measure_while_training', action='store_true', default=False, help='disables measure while training (make program faster)')
    parser.add_argument('--batch_size', type=int, default=256, help="Number of batches to train/test for. Default: 256")
    parser.add_argument('--dataset', type=int, default=7)
    parser.add_argument('--nocuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--results_dir', default='/home/zhihuan/Documents/SALMON/experiments/Results', help="results dir")
    return parser.parse_args()

if __name__=='__main__':
    torch.cuda.empty_cache()
    args = parse_args()
    plt.ioff()

    # model file
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate_range = 10**np.arange(-4,-1,0.3)
    cuda = True
    verbose = 0
    measure_while_training = True
    dropout_rate = 0
    lambda_1 = 1e-5 # L1
    
    

    if args.dataset == 1:
        dataset_subset = "1_RNAseq"
    elif args.dataset == 2:
        dataset_subset = "2_miRNAseq"
    elif args.dataset == 3:
        dataset_subset = "3_RNAseq+miRNAseq"
    elif args.dataset == 4:
        dataset_subset = "4_RNAseq+miRNAseq+cnv+tmb"
    elif args.dataset == 5:
        dataset_subset = "5_RNAseq+miRNAseq+clinical"
    elif args.dataset == 6:
        dataset_subset = "6_cnv+tmb+clinical"
    elif args.dataset == 7:
        dataset_subset = "7_RNAseq+miRNAseq+cnv+tmb+clinical"
        
    datasets_5folds = pickle.load( open( '/home/zhihuan/Documents/SALMON/data/BRCA_583_new/datasets_5folds.pickle', "rb" ) )

        
    for i in range(5):
        print("5 fold CV -- %d/5" % (i+1))
        
        # dataset
        TIMESTRING  = time.strftime("%Y%m%d-%H.%M.%S", time.localtime())
        
        results_dir_dataset = args.results_dir + '/' + dataset_subset + '/run_' + TIMESTRING + '_fold_' + str(i+1)
        if not os.path.exists(results_dir_dataset):
            os.makedirs(results_dir_dataset)
            
        logging.basicConfig(filename=results_dir_dataset+'/mainlog.log',level=logging.DEBUG)
    #    print("Arguments:",args)
    #    logging.info("Arguments: %s" % args)
        datasets = datasets_5folds[str(i+1)]
        
        len_of_RNAseq = 57
        len_of_miRNAseq = 12
        len_of_cnv = 1
        len_of_tmb = 1
        len_of_clinical = 3
        
        length_of_data = {}
        length_of_data['mRNAseq'] = len_of_RNAseq
        length_of_data['miRNAseq'] = len_of_miRNAseq
        length_of_data['CNB'] = len_of_cnv
        length_of_data['TMB'] = len_of_tmb
        length_of_data['clinical'] = len_of_clinical
        
        if args.dataset == 1:
            ####      RNAseq Only
            datasets['train']['x'] = datasets['train']['x'][:, 0:len_of_RNAseq]
            datasets['test']['x'] = datasets['test']['x'][:, 0:len_of_RNAseq]
        elif args.dataset == 2:
            ####     miRNAseq Only
            datasets['train']['x'] = datasets['train']['x'][:, len_of_RNAseq:(len_of_RNAseq + len_of_miRNAseq)]
            datasets['test']['x'] = datasets['test']['x'][:, len_of_RNAseq:(len_of_RNAseq + len_of_miRNAseq)]
        elif args.dataset == 3:
            ####      RNAseq + miRNAseq
            datasets['train']['x'] = datasets['train']['x'][:, 0:(len_of_RNAseq + len_of_miRNAseq)]
            datasets['test']['x'] = datasets['test']['x'][:, 0:(len_of_RNAseq + len_of_miRNAseq)]
        elif args.dataset == 4:
            ####      RNAseq + miRNAseq + CNB + all TMB
            datasets['train']['x'] = datasets['train']['x'][:, 0:(len_of_RNAseq + len_of_miRNAseq + len_of_cnv + len_of_tmb)]
            datasets['test']['x'] = datasets['test']['x'][:, 0:(len_of_RNAseq + len_of_miRNAseq + len_of_cnv + len_of_tmb)]
        elif args.dataset == 5:
            ####      RNAseq + miRNAseq + clinical (age+ER+PR)
            datasets['train']['x'] = np.concatenate((datasets['train']['x'][:, 0:(len_of_RNAseq + len_of_miRNAseq)], \
                                        datasets['train']['x'][:, (len_of_RNAseq + len_of_miRNAseq + len_of_cnv + len_of_tmb):]),1)
            datasets['test']['x'] = np.concatenate((datasets['test']['x'][:, 0:(len_of_RNAseq + len_of_miRNAseq)], \
                                        datasets['test']['x'][:, (len_of_RNAseq + len_of_miRNAseq + len_of_cnv + len_of_tmb):]),1)
        elif args.dataset == 6:
            ####      CNB + all TMB + clinical (age+ER+PR)
            datasets['train']['x'] = datasets['train']['x'][:, (len_of_RNAseq + len_of_miRNAseq):]
            datasets['test']['x'] = datasets['test']['x'][:, (len_of_RNAseq + len_of_miRNAseq):]

        elif args.dataset == 7:
            ####      RNAseq + miRNAseq + CNB + all TMB + clinical (age+ER+PR)
            datasets['train']['x'] = datasets['train']['x']
            datasets['test']['x'] = datasets['test']['x']
    
    # =============================================================================
    # # Finding optimal learning rate w.r.t. concordance index
    # =============================================================================
        ci_list = []
        for j, lr in enumerate(learning_rate_range):
            print("[%d/%d] current lr: %.4E" %((j+1), len(learning_rate_range), lr))
            logging.info("[%d/%d] current lr: %.4E" %((j+1), len(learning_rate_range), lr))
            model, loss_nn_all, pvalue_all, c_index_all, c_index_list, acc_train_all, code_output = \
                 SALMON.train(datasets, num_epochs, batch_size, lr, dropout_rate,\
                                         lambda_1, length_of_data, cuda, measure_while_training, verbose)
        
            epochs_list = range(num_epochs)
            plt.figure(figsize=(8,4))
            plt.plot(epochs_list, c_index_list['train'], "b--",linewidth=1)
            plt.plot(epochs_list, c_index_list['test'], "g-",linewidth=1)
            plt.legend(['train', 'test'])
            plt.xlabel("epochs")
            plt.ylabel("Concordance index")
            plt.savefig(results_dir_dataset + "/convergence_%02d_lr=%.2E.png" % (j, lr),dpi=300)
            
            code_test, loss_nn_sum, acc_test, pvalue_pred, c_index_pred, lbl_pred_all, OS_event_test, OS_test = \
                SALMON.test(model, datasets, 'test', length_of_data, batch_size, cuda, verbose)
            ci_list.append(c_index_pred)
            print("current concordance index: ", c_index_pred,"\n")
            logging.info("current concordance index: %.10f\n" % c_index_pred)
            
        optimal_lr = learning_rate_range[np.argmax(ci_list)]
        
        print("Optimal learning rate: %.4E, optimal c-index: %.10f" % (optimal_lr, np.max(ci_list)))
        logging.info("Optimal learning rate: %.4E, optimal c-index: %.10f" % (optimal_lr, np.max(ci_list)))
    
    
    # =============================================================================
    # # Training 
    # =============================================================================
    
        model, loss_nn_all, pvalue_all, c_index_all, c_index_list, acc_train_all, code_output = \
                 SALMON.train(datasets, num_epochs, batch_size, optimal_lr, dropout_rate,\
                                         lambda_1, length_of_data, cuda, measure_while_training, verbose)
        code_train, loss_nn_sum, acc_train, pvalue_pred, c_index_pred, lbl_pred_all_train, OS_event_train, OS_train = \
            SALMON.test(model, datasets, 'train', length_of_data, batch_size, cuda, verbose)
        print("[Final] Apply model to training set: c-index: %.10f, p-value: %.10e" % (c_index_pred, pvalue_pred))
        logging.info("[Final] Apply model to training set: c-index: %.10f, p-value: %.10e" % (c_index_pred, pvalue_pred))
    
        code_test, loss_nn_sum, acc_test, pvalue_pred, c_index_pred, lbl_pred_all_test, OS_event_test, OS_test = \
            SALMON.test(model, datasets, 'test', length_of_data, batch_size, cuda, verbose)
        print("[Final] Apply model to testing set: c-index: %.10f, p-value: %.10e" % (c_index_pred, pvalue_pred))
        logging.info("[Final] Apply model to testing set: c-index: %.10f, p-value: %.10e" % (c_index_pred, pvalue_pred))
               
        
        with open(results_dir_dataset + '/model.pickle', 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(results_dir_dataset + '/c_index_list_by_epochs.pickle', 'wb') as handle:
            pickle.dump(c_index_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        with open(results_dir_dataset + '/hazard_ratios_lbl_pred_all_train.pickle', 'wb') as handle:
            pickle.dump(lbl_pred_all_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(results_dir_dataset + '/OS_event_train.pickle', 'wb') as handle:
            pickle.dump(OS_event_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(results_dir_dataset + '/OS_train.pickle', 'wb') as handle:
            pickle.dump(OS_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(results_dir_dataset + '/code_train.pickle', 'wb') as handle:
            pickle.dump(code_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        with open(results_dir_dataset + '/hazard_ratios_lbl_pred_all_test.pickle', 'wb') as handle:
            pickle.dump(lbl_pred_all_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(results_dir_dataset + '/OS_event_test.pickle', 'wb') as handle:
            pickle.dump(OS_event_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(results_dir_dataset + '/OS_test.pickle', 'wb') as handle:
            pickle.dump(OS_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(results_dir_dataset + '/code_test.pickle', 'wb') as handle:
            pickle.dump(code_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
        epochs_list = range(num_epochs)
        plt.figure(figsize=(8,4))
        plt.plot(epochs_list, c_index_list['train'], "b--",linewidth=1)
        plt.plot(epochs_list, c_index_list['test'], "g-",linewidth=1)
        plt.legend(['train', 'test'])
        plt.xlabel("epochs")
        plt.ylabel("Concordance index")
        plt.savefig(results_dir_dataset + "/convergence.png",dpi=300)
        


