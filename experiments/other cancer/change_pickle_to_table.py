#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 17:05:50 2019

@author: zhihuan
"""


import os
import pandas as pd
import pickle


workdir = '/home/zhihuan/Documents/SALMON/experiments/Results/LUAD/'

result_dirs = [workdir + r + '/' for r in os.listdir(workdir) if os.path.isdir(workdir + r)]


for result_dir in result_dirs:
    print(result_dir)
    fold_dirs = [result_dir + r + '/' for r in os.listdir(result_dir) if os.path.isdir(result_dir + r)]
    
    for f_d in fold_dirs:
        with open(f_d + '/hazard_ratios_lbl_pred_all_test.pickle', 'rb') as f:
            file = pd.DataFrame(pickle.load(f))
            file.to_csv(f_d + '/hazard_ratios_lbl_pred_all_test.csv')
            
        with open(f_d + '/hazard_ratios_lbl_pred_all_train.pickle', 'rb') as f:
            file = pd.DataFrame(pickle.load(f))
            file.to_csv(f_d + '/hazard_ratios_lbl_pred_all_train.csv')
            
        with open(f_d + '/OS_event_test.pickle', 'rb') as f:
            file = pd.DataFrame(pickle.load(f))
            file.to_csv(f_d + '/OS_event_test.csv')
            
        with open(f_d + '/OS_event_train.pickle', 'rb') as f:
            file = pd.DataFrame(pickle.load(f))
            file.to_csv(f_d + '/OS_event_train.csv')
            
        with open(f_d + '/OS_test.pickle', 'rb') as f:
            file = pd.DataFrame(pickle.load(f))
            file.to_csv(f_d + '/OS_test.csv')
            
        with open(f_d + '/OS_train.pickle', 'rb') as f:
            file = pd.DataFrame(pickle.load(f))
            file.to_csv(f_d + '/OS_train.csv')