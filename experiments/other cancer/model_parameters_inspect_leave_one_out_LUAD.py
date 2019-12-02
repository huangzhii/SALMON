#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Zhi Huang
"""
import sys, os
sys.path.append("/home/zhihuan/Documents/SALMON/model")
import SALMON
import os, sys
import pickle
import numpy as np
from itertools import repeat
import matplotlib.pyplot as plt
import pandas as pd
import copy
from sklearn.model_selection import KFold
from sklearn import preprocessing

result_dir = "/home/zhihuan/Documents/SALMON/experiments/Results/LUAD/7_RNAseq+miRNAseq+cnb+tmb+clinical"
files = os.listdir(result_dir)
files = [s for s in files if "run" in s]
files = sorted(files)

# =============================================================================
# Dataset
# =============================================================================
dataset_dir = '/home/zhihuan/Documents/SALMON/data/LUAD/multiomics_preprocessing_results/'
# 5-fold data
tempdata = {}
tempdata['clinical'] = pd.read_csv(dataset_dir + 'clinical.csv', index_col = 0).reset_index(drop = True)
tempdata['mRNAseq_eigengene'] = pd.read_csv(dataset_dir + 'mRNAseq_eigengene_matrix.csv', index_col = 0).reset_index(drop = True)
tempdata['miRNAseq_eigengene'] = pd.read_csv(dataset_dir + 'miRNAseq_eigengene_matrix.csv', index_col = 0).reset_index(drop = True)
tempdata['TMB'] = pd.read_csv(dataset_dir + 'TMB.csv', index_col = 0).reset_index(drop = True)
tempdata['CNB'] = pd.read_csv(dataset_dir + 'CNB.csv', index_col = 0).reset_index(drop = True)
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
    datasets_5folds[ix]['test']['x'] = data['x'][test_index, :]
    datasets_5folds[ix]['test']['e'] = data['e'][test_index]
    datasets_5folds[ix]['test']['t'] = data['t'][test_index]


length_of_data = {}
length_of_data['mRNAseq'] = tempdata['mRNAseq_eigengene'].shape[1]
length_of_data['miRNAseq'] = tempdata['miRNAseq_eigengene'].shape[1]
length_of_data['CNB'] = 1
length_of_data['TMB'] = 1
length_of_data['clinical'] = 2





for fold in range(len(files)):
    fname = result_dir+"/"+files[fold]+"/leave_one_out_CIndex.txt"
    open(fname, "w").close()
    f = open(fname,"w+")
    datasets = datasets_5folds[fold+1]
    batch_size, cuda, verbose = 128, True, 0
    ci_list = []
    model = pickle.load( open(result_dir+"/"+files[fold]+'/model.pickle', "rb" ) )
    code_test, loss_nn_sum, acc_test, pvalue_pred, c_index_pred, lbl_pred_all_test, OS_event_test, OS_test = \
        SALMON.test(model, datasets, 'test', length_of_data, batch_size, cuda, verbose)
    print("Without feature removing concordance index: %.8f\n" % c_index_pred)
    f.write("Without feature removing concordance index: %.8f\n" % c_index_pred)
    for i in range(datasets['test']['x'].shape[1]):
        current_feature_name = data['column_names'][i]
        datasets_leave_one = copy.deepcopy(datasets)
        datasets_leave_one['test']['x'][:,i] = 0
        code_test, loss_nn_sum, acc_test, pvalue_pred, c_index_pred, lbl_pred_all_test, OS_event_test, OS_test = \
            SALMON.test(model, datasets_leave_one, 'test', length_of_data, batch_size, cuda, verbose)
        print("Feature removed: %d, name: %s, concordance index: %.8f\n" % (i+1, current_feature_name, c_index_pred))
        f.write("Feature removed: %d, name: %s, concordance index: %.8f\n" % (i+1, current_feature_name, c_index_pred))

    f.close()
