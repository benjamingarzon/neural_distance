#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 10:49:47 2021

@author: benjamin.garzon@gmail.com
"""
from random import choice
from config import TRAIN_FILE, TEST_FILE, TARGET_COL, PARAM_GRID
from util import create_pairs, plot_embedding, loop_params, balance_labels
import pandas as pd
from models import SiameseNet
import numpy as np 
import matplotlib.pyplot as plt

train_data = pd.read_csv(TRAIN_FILE)
test_data = pd.read_csv(TEST_FILE)
feature_cols = [ x for x in train_data.columns if 'feature' in x ]
y_train = train_data[TARGET_COL].values
X_train = train_data[feature_cols].values
trained_labels = np.unique(y_train)
y_test = test_data[TARGET_COL]
# remove labels that were not trained
X_test = test_data[feature_cols]
X_test = X_test.loc[y_test.isin(trained_labels), :].values
y_test = y_test.loc[y_test.isin(trained_labels)].values

def test_pairs():
    X = X_train
    y = y_train
    index_pairs, X_pairs, label_pairs = create_pairs(X, y)
    
    index = choice(range(len(index_pairs)))
    i, j = index_pairs[index]

    assert 1*(y[i] == y[j]) == label_pairs[index]
    assert len(label_pairs) == len(index_pairs)
    
    return(index_pairs, X_pairs, label_pairs)

def test_siamese():
    
    index_train, X_pairs_train, label_train = create_pairs(X_train, y_train)
    index_test, X_pairs_test, label_test = create_pairs(X_test, y_test)
    net = SiameseNet(TARGET_COL)
    net.build(input_shape = (X_train.shape[1]) )
    #net.fit(X_pairs_train, label_train, X_pairs_test, label_test)
    return net

def test_loop_params():
    return loop_params(PARAM_GRID, sample = None)
    
def test_fit_and_predict_cv():
    filename = '/data/lv0/MotorSkill/fmriprep/analysis/roi_data/sub-lue1101_ses-4_roi_data_mask_rh_R_SPL.csv'
    net = SiameseNet(TARGET_COL)
    net.fit_and_predict_file_cv(filename)

def test_balance_labels():
    print(balance_labels(train_data, TARGET_COL))


#plt.hist(X_train.flatten())
#print(np.std(X_train.flatten()))
#print(np.mean(X_train.flatten()))

test_siamese()
#print(len(test_loop_params()))
#test_fit_and_predict_cv()
#test_balance_labels()
#index_pairs, X_pairs, label_pairs = test_pairs()
#net = test_siamese()

#embedding_train = net.predict_embedding(X_train)
#plot_embedding(embedding_train, y_train)
#embedding_test = net.predict_embedding(X_test)
#plot_embedding(embedding_test, y_test)
