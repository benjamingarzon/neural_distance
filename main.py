#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 13:07:01 2021

@author: benjamin.garzon@gmail.com
"""
from config import FILES_DIR, TARGET_COL, TEST_LABEL, PARAM_GRID, NUM_CORES, \
    OUTPUT_DIR, DEFAULT_SIAMESE_PARAMS
from models import SiameseNet
from distance_estimator import Estimator
import numpy as np
import os
from util import loop_params, plot_results
from joblib import Parallel, delayed
import pickle
import random 

shuffle = False
if shuffle:
    results_file = os.path.join(OUTPUT_DIR, 'results_shuffled.pkl')
else:
    results_file = os.path.join(OUTPUT_DIR, 'results.pkl')

def launch_experiment(files_dir, target_col, ModelClass, params = None, 
                      shuffle = False, return_scores = False):

    estimator = Estimator(files_dir, target_col, ModelClass, params, shuffle = shuffle)
    estimator.parse_files()
    # use only trainers in first wave and later sessions
    mysubjects = [x for x in estimator.subjects if 'lue11' in x ]
    random.shuffle(mysubjects)
    selected_sessions = list(range(4, 8))
    scores, mean_score = estimator.estimate(subjects = mysubjects, labels = TEST_LABEL, 
                                selected_sessions = selected_sessions)
    # for now aggregate across sessions
    if return_scores:
        return scores, mean_score
    else:
        return mean_score

# make experiments

#if __name__ == '__main__':
if True:                                         

    mean_score = launch_experiment(FILES_DIR, 
                               TARGET_COL, 
                               SiameseNet, 
                               DEFAULT_SIAMESE_PARAMS, 
                               shuffle, 
                               return_scores=True)
else:
    params_list = loop_params(PARAM_GRID)
    score_list = Parallel(n_jobs = NUM_CORES, 
    #                      require = "sharedmem", 
                          verbose = 0)(delayed(launch_experiment) 
                                             (FILES_DIR, TARGET_COL, SiameseNet, 
                                              params, shuffle)
                                              for params in params_list)
    
    results = {'score_list' : score_list, 'params_list' : params_list}
    best_score = max(results['score_list'])
    best_index = results['score_list'].index(best_score)
    best_params = results['params_list'][best_index]
    print(best_score)
    print(best_params)

    with open(results_file, 'wb') as output:
        pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)   
        
plot_results(results_file)

