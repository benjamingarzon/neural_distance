#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 15:31:50 2021

@author: benjamin.garzon@gmail.com
"""
from config import OUTPUT_DIR
import os
from util import plot_results
import pickle

results_file = os.path.join(OUTPUT_DIR, 'results.pkl')

with open(results_file, 'rb') as f:
    results = pickle.load(f)   
params = results['best_params']
plot(results['best_score'])
plot_results(results_file)


