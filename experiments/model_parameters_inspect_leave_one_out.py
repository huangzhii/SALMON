#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Zhi Huang
"""
import sys, os
sys.path.append("/home/zhihuan/Documents/SALMON")
import SALMON
import os, sys
import pickle
import numpy as np
from itertools import repeat
import matplotlib.pyplot as plt
import pandas as pd
import copy
result_dir = "/home/zhihuan/Documents/SALMON/experiments/Results/7_RNAseq+miRNAseq+cnv+tmb+clinical"
files = os.listdir(result_dir)
files = [s for s in files if "run" in s]
files = sorted(files)
datasets_5folds = pickle.load( open( '/home/zhihuan/Documents/SALMON/data/BRCA_583_new/datasets_5folds.pickle', "rb" ) )
for fold in range(len(files)):
    fname = result_dir+"/"+files[fold]+"/leave_one_out_CIndex.txt"
    open(fname, "w").close()
    f = open(fname,"w+")
    datasets = datasets_5folds[str(fold+1)]
    len_of_RNAseq = 57
    len_of_miRNAseq = 12
    len_of_cnv = 1
    len_of_tmb = 1
    len_of_clinical = 3
    batch_size, cuda, verbose = 128, True, 0
    ci_list = []
    model = pickle.load( open(result_dir+"/"+files[fold]+'/model.pickle', "rb" ) )
    code_test, loss_nn_sum, acc_test, pvalue_pred, c_index_pred, lbl_pred_all_test, OS_event_test, OS_test = \
        SALMON.test(model, datasets, 'test', batch_size, cuda, verbose)
    print("Without feature removing concordance index: %.8f\n" % c_index_pred)
    f.write("Without feature removing concordance index: %.8f\n" % c_index_pred)
    for i in range(datasets['test']['x'].shape[1]):
        datasets_leave_one = copy.deepcopy(datasets)
        datasets_leave_one['test']['x'][:,i] = 0
        code_test, loss_nn_sum, acc_test, pvalue_pred, c_index_pred, lbl_pred_all_test, OS_event_test, OS_test = \
            SALMON.test(model, datasets_leave_one, 'test', batch_size, cuda, verbose)
        print("Feature removed: %d, concordance index: %.8f\n" % (i+1, c_index_pred))
        f.write("Feature removed: %d, concordance index: %.8f\n" % (i+1, c_index_pred))

    f.close()
