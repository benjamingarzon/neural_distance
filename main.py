#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 13:07:01 2021

@author: benjamin.garzon@gmail.com
"""
from config import FILES_DIR, TARGET_COL, TEST_LABEL, PARAM_GRID, NUM_CORES, \
    OUTPUT_DIR, DEFAULT_SIAMESE_PARAMS, SCORES_PATH, MODEL_PATH, SHUFFLE, \
        PLOT_PATH, PARAMETER_TUNING, BLOCK_SIZE
from models import SiameseNet, TripletNet
from distance_estimator import Estimator
import numpy as np
import os
from util import loop_params, plot_results, plot_scores
from joblib import Parallel, delayed
import pickle
import random 
import pandas as pd

if SHUFFLE:
    results_file = os.path.join(OUTPUT_DIR, 'results_shuffled.pkl')
else:
    results_file = os.path.join(OUTPUT_DIR, 'results.pkl')

def launch_experiment(files_dir, target_col, params, 
                      shuffle = False, return_scores = False, tuning = False):
    if params['modelclass'] == 'SiameseNet_cont':
        ModelClass = SiameseNet
        params['epochs'] = 30
        params['loss'] = 'contrastive_loss'
        
    if params['modelclass'] == 'SiameseNet_bin':
        ModelClass = SiameseNet
        params['epochs'] = 30
        params['loss'] = 'binary_crossentropy'
        
    if params['modelclass'] == 'TripletNet':
        ModelClass = TripletNet
        params['epochs'] = 5
        params['loss'] = 'triplet_loss'
        
    estimator = Estimator(files_dir, target_col, ModelClass, params, 
                          shuffle = shuffle, 
                          grouping_variable = 'seq_train')

    estimator.parse_files()
    if tuning:
        # use only first wave with later sessions
        mysubjects = [x for x in estimator.subjects if 'lue1107' in x ]
        selected_sessions = list(range(3, 8))
    else:
        mysubjects = [x for x in estimator.subjects if 'lue1' not in x ]

#    random.shuffle(mysubjects)
    scores, mean_score = estimator.estimate(subjects = mysubjects, labels = TEST_LABEL, 
                                selected_sessions = selected_sessions)
    # for now aggregate across sessions
    if return_scores:
        return scores, mean_score
    else:
        return mean_score

# make experiments

#if __name__ == '__main__':
if PARAMETER_TUNING:                                         
    os.system('rm -r %s'%os.path.join(MODEL_PATH, 'exp*'))
    os.system('rm -r %s'%os.path.join(PLOT_PATH, '*.png'))

    params_list = loop_params(PARAM_GRID)
    params_len = len(params_list)
    block_index = 0
    score_list = []
    while True:
        max_index = min(block_index + BLOCK_SIZE, params_len)
            
        params_block = params_list[block_index:max_index]
        block_index = block_index + BLOCK_SIZE
        score_block = Parallel(n_jobs = NUM_CORES, 
        #                      require = "sharedmem", 
                              verbose = 0)(delayed(launch_experiment) 
                                                 (FILES_DIR, TARGET_COL, 
                                                  params, SHUFFLE, tuning = True)
                                                  for params in params_block)
        score_list.extend(score_block)

        # Update results
        results = {'score_list' : score_list, 'params_list' : params_list}
        results['best_score'] = max(results['score_list'])
        results['best_index'] = results['score_list'].index(results['best_score'])
        results['best_params'] = results['params_list'][results['best_index']]
    
        print("Best parameter configuration so far")
        print(results['best_score'])
        print(results['best_params'])
    
        with open(results_file, 'wb') as output:
            pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)   

        if max_index == params_len:
            break
    plot_results(results_file)


else:
    try:
        with open(results_file, 'rb') as f:
            results = pickle.load(f)   
        params = results['best_params']
    except FileNotFoundError:
        params = DEFAULT_SIAMESE_PARAMS
        
scores, mean_score = launch_experiment(FILES_DIR, 
                           TARGET_COL, 
                           params, 
                           shuffle = False, 
                           return_scores=True)

scores_shuffled, mean_score_shuffled = launch_experiment(FILES_DIR, 
                           TARGET_COL, 
                           params, 
                           shuffle = True, 
                           return_scores=True)

scores['shuffled'] = 0
scores_shuffled['shuffled'] = 1
allscores = pd.concat(scores, scores_shuffled)
allscores.to_csv(SCORES_PATH)
plot_scores(allscores)

