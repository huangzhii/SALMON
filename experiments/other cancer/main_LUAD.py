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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='/home/zhihuan/Documents/SALMON/data/LUAD/multiomics_preprocessing_results/', help="datasets")
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs to train for. Default: 100")
    parser.add_argument('--measure_while_training', action='store_true', default=False, help='disables measure while training (make program faster)')
    parser.add_argument('--batch_size', type=int, default=64, help="Number of batches to train/test for. Default: 256")
    parser.add_argument('--dataset', type=int, default=7)
    parser.add_argument('--nocuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--results_dir', default='/home/zhihuan/Documents/SALMON/experiments/Results/LUAD', help="results dir")
    return parser.parse_args()

if __name__=='__main__':
    torch.cuda.empty_cache()
    args = parse_args()

    # model file
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate_range = 10**np.arange(-4,-1,0.3)
    cuda = True
    verbose = 0
    measure_while_training = True
    dropout_rate = 0
    lambda_1 = 1e-6 # L1
    
    # 5-fold data
    tempdata = {}
    tempdata['clinical'] = pd.read_csv(args.dataset_dir + 'clinical.csv', index_col = 0).reset_index(drop = True)
    tempdata['mRNAseq_eigengene'] = pd.read_csv(args.dataset_dir + 'mRNAseq_eigengene_matrix.csv', index_col = 0).reset_index(drop = True)
    tempdata['miRNAseq_eigengene'] = pd.read_csv(args.dataset_dir + 'miRNAseq_eigengene_matrix.csv', index_col = 0).reset_index(drop = True)
    tempdata['TMB'] = pd.read_csv(args.dataset_dir + 'TMB.csv', index_col = 0).reset_index(drop = True)
    tempdata['CNB'] = pd.read_csv(args.dataset_dir + 'CNB.csv', index_col = 0).reset_index(drop = True)
    tempdata['CNB']['log2_LENGTH_KB'] = np.log2(tempdata['CNB']['LENGTH_KB'].values + 1)
    
    print('0:MALE\t\t1:FEMALE\n0:Alive\t\t1:Dead')
    tempdata['clinical']['gender'] = (tempdata['clinical']['gender'].values == 'MALE').astype(int)
    tempdata['clinical']['vital_status'] = (tempdata['clinical']['vital_status'].values == 'Dead').astype(int)
    
    
    data = {}
    data['x'] = pd.concat((tempdata['mRNAseq_eigengene'], tempdata['miRNAseq_eigengene'], tempdata['CNB']['log2_LENGTH_KB'], tempdata['TMB']['All_TMB'], tempdata['clinical'][['gender','age_at_initial_pathologic_diagnosis']]), axis = 1).values.astype(np.double)
    all_column_names = ['mRNAseq_' + str(i+1) for i in range(tempdata['mRNAseq_eigengene'].shape[1])] + \
                            ['miRNAseq_' + str(i+1) for i in range(tempdata['miRNAseq_eigengene'].shape[1])] + \
                            ['CNB', 'TMB', 'GENDER', 'AGE']
    print('perform min-max scaler on all input features')
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(data['x'])
    data['x'] = scaler.transform(data['x'])
    
    data['e'] = tempdata['clinical']['vital_status'].values.astype(np.int32)
    data['t'] = tempdata['clinical']['survival_days'].values.astype(np.double)
    
    if args.dataset == 1:
        dataset_subset = "1_RNAseq"
        data['column_names'] = ['mRNAseq_' + str(i+1) for i in range(tempdata['mRNAseq_eigengene'].shape[1])]
        
    elif args.dataset == 2:
        dataset_subset = "2_miRNAseq"
        data['column_names'] = ['miRNAseq_' + str(i+1) for i in range(tempdata['miRNAseq_eigengene'].shape[1])]
        
    elif args.dataset == 3:
        dataset_subset = "3_RNAseq+miRNAseq"
        data['column_names'] = ['mRNAseq_' + str(i+1) for i in range(tempdata['mRNAseq_eigengene'].shape[1])] + \
                                ['miRNAseq_' + str(i+1) for i in range(tempdata['miRNAseq_eigengene'].shape[1])]
    elif args.dataset == 4:
        dataset_subset = "4_RNAseq+miRNAseq+cnb+tmb"
        data['column_names'] = ['mRNAseq_' + str(i+1) for i in range(tempdata['mRNAseq_eigengene'].shape[1])] + \
                                ['miRNAseq_' + str(i+1) for i in range(tempdata['miRNAseq_eigengene'].shape[1])] + \
                                ['CNB', 'TMB']
    elif args.dataset == 5:
        dataset_subset = "5_RNAseq+miRNAseq+clinical"
        data['column_names'] = ['mRNAseq_' + str(i+1) for i in range(tempdata['mRNAseq_eigengene'].shape[1])] + \
                                ['miRNAseq_' + str(i+1) for i in range(tempdata['miRNAseq_eigengene'].shape[1])] + \
                                ['GENDER', 'AGE']
    elif args.dataset == 6:
        dataset_subset = "6_cnb+tmb+clinical"
        data['column_names'] = ['CNB', 'TMB', 'GENDER', 'AGE']
        
    elif args.dataset == 7:
        dataset_subset = "7_RNAseq+miRNAseq+cnb+tmb+clinical"
        data['column_names'] = ['mRNAseq_' + str(i+1) for i in range(tempdata['mRNAseq_eigengene'].shape[1])] + \
                                ['miRNAseq_' + str(i+1) for i in range(tempdata['miRNAseq_eigengene'].shape[1])] + \
                                ['CNB', 'TMB', 'GENDER', 'AGE']
    print('subsetting data...')
    data['x'] = data['x'][:, [i for i, c in enumerate(all_column_names) if c in data['column_names']]]

    kf = KFold(n_splits=5, shuffle=True, random_state=666)
    datasets_5folds = {}
    for ix, (train_index, test_index) in enumerate(kf.split(data['x']), start = 1):
        datasets_5folds[ix] = {}
        datasets_5folds[ix]['train'] = {}
        datasets_5folds[ix]['train']['x'] = data['x'][train_index, :]
        datasets_5folds[ix]['train']['e'] = data['e'][train_index]
        datasets_5folds[ix]['train']['t'] = data['t'][train_index]
        datasets_5folds[ix]['test'] = {}
        datasets_5folds[ix]['test']['x'] = data['x'][train_index, :]
        datasets_5folds[ix]['test']['e'] = data['e'][train_index]
        datasets_5folds[ix]['test']['t'] = data['t'][train_index]

    for i in range(1, len(datasets_5folds) + 1):
        print("5 fold CV -- %d/5" % i)
        
        # dataset
        TIMESTRING  = time.strftime("%Y%m%d-%H.%M.%S", time.localtime())
        
        results_dir_dataset = args.results_dir + '/' + dataset_subset + '/run_' + TIMESTRING + '_fold_' + str(i)
        if not os.path.exists(results_dir_dataset):
            os.makedirs(results_dir_dataset)
            
        logging.basicConfig(filename=results_dir_dataset+'/mainlog.log',level=logging.DEBUG)
    #    print("Arguments:",args)
    #    logging.info("Arguments: %s" % args)
        datasets = datasets_5folds[i]
        
        length_of_data = {}
        length_of_data['mRNAseq'] = tempdata['mRNAseq_eigengene'].shape[1]
        length_of_data['miRNAseq'] = tempdata['miRNAseq_eigengene'].shape[1]
        length_of_data['CNB'] = 1
        length_of_data['TMB'] = 1
        length_of_data['clinical'] = 2
        
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
            plt.close()
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
        plt.close()


