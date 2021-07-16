#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 13:07:01 2021

@author: benjamin.garzon@gmail.com
"""
from config import FILES_DIR, TARGET_COL, TEST_LABEL, PARAM_GRID, NUM_CORES, \
    OUTPUT_DIR
from models import SiameseNet
from distance_estimator import Estimator
import numpy as np
import os
from util import loop_params
from joblib import Parallel, delayed

#if __name__ == '__main__':
def launch_experiment(files_dir, target_col, ModelClass, params = None):

    estimator = Estimator(files_dir, target_col, ModelClass, params)
    estimator.parse_files()
    # use only trainers in first wave and later sessions
    mysubjects = [x for x in estimator.subjects if 'lue11' in x ]
    selected_sessions = list(range(4, 8))
    scores, mean_score = estimator.estimate(subjects = mysubjects, labels = TEST_LABEL, 
                                selected_sessions = selected_sessions)
    # for now aggregate across sessions
    return mean_score

#mean_score = launch_experiment(FILES_DIR, TARGET_COL, SiameseNet)
#print(mean_score)
# make experiments

params_list = loop_params(PARAM_GRID, sample = 300)
score_list = Parallel(n_jobs = NUM_CORES, 
#                      require = "sharedmem", 
                      verbose = 0)(delayed(launch_experiment) 
                                         (FILES_DIR, TARGET_COL, SiameseNet, 
                                          params)
                                          for params in params_list)

best_score = max(score_list)
best_index = score_list.index(best_score)
best_params = params_list[best_index]
print(best_score)
print(best_params)
results = {'score_list' : score_list, 'params_list' : params_list}

import pickle
with open(os.path.join(OUTPUT_DIR, 'results.pkl'), 'wb') as output:
    pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)