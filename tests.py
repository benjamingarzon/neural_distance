#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 10:49:47 2021

@author: benjamin.garzon@gmail.com
"""
from random import choice
from config import TRAIN_FILE, TEST_FILE, TARGET_COL, PARAM_GRID, PLOT_PATH
from util import create_pairs, create_triplets, plot_embedding, loop_params, \
    balance_labels, metrics
import pandas as pd
from models import SiameseNet, TripletNet
import numpy as np 
import matplotlib.pyplot as plt
import os

train_data = pd.read_csv(TRAIN_FILE)
test_data = pd.read_csv(TEST_FILE)
feature_cols = [ x for x in train_data.columns if 'feature' in x ]
y_train = train_data[TARGET_COL].values
X_train = train_data[feature_cols].values
trained_labels = np.unique(y_train)
# remove labels that were not trained
test_data = test_data.loc[test_data[TARGET_COL].isin(trained_labels)]

X_test = test_data[feature_cols].values 
y_test = test_data[TARGET_COL].values


def test_pairs():
    X = X_train
    y = y_train
    index_pairs, X_pairs, label_pairs = create_pairs(X, y)
    
    index = choice(range(len(index_pairs)))
    i, j = index_pairs[index]

    assert 1*(y[i] == y[j]) == label_pairs[index]
    assert len(label_pairs) == len(index_pairs)
  
    return(index_pairs, X_pairs, label_pairs)

def test_triplets():
    X = X_train
    y = y_train
    index_triplets, X_triplets = create_triplets(X, y, sample = None)
    
    index = choice(range(len(index_triplets)))
    i, j, k = index_triplets[index]
    print(y[i], '\n', y[j], '\n', y[k])
    assert y[i] == y[j]
    assert y[i] != y[k]
  
    return(index_triplets, X_triplets)

def test_siamese():
    
    index_train, X_pairs_train, label_train = create_pairs(X_train, y_train)
    index_test, X_pairs_test, label_test = create_pairs(X_test, y_test)
    net = SiameseNet(TARGET_COL)
    net.build(input_shape = (X_train.shape[1]) )
    net.fit(X_pairs_train, label_train, X_pairs_test, label_test)
    return net

def test_tripletnet():
    s = None
    index_train, X_tuples_train, label_train = create_triplets(X_train, y_train, sample = s)
    index_test, X_tuples_test, label_test = create_triplets(X_test, y_test)
    net = TripletNet(TARGET_COL)
    net.build(input_shape = (X_train.shape[1]) )
    net.fit(X_tuples_train, label_train, X_tuples_test, label_test)
    return net

def test_loop_params():
    return loop_params(PARAM_GRID, sample = None)
    
def test_fit_and_predict_cv():
    filename = '/data/lv0/MotorSkill/fmriprep/analysis/roi_data/sub-lue1101_ses-4_roi_data_mask_rh_R_SPL.csv'
    net = SiameseNet(TARGET_COL)
    net.fit_and_predict_file_cv(filename)

def test_balance_labels():
    print(balance_labels(train_data, TARGET_COL))

def test_metrics():
    embedding_test = net.predict_embedding(X_test)
    grouping_labels = test_data['seq_train'].values
    mymetrics = metrics(embedding_test, X_test, y_test,
                        grouping_labels = grouping_labels, plot = True)


#plt.hist(X_train.flatten())
#print(np.std(X_train.flatten()))
#print(np.mean(X_train.flatten()))
#net = test_tripletnet()
net = test_siamese()
#stophere
#print(len(test_loop_params()))
#test_fit_and_predict_cv()
#test_balance_labels()
#index_pairs, X_pairs, label_pairs = test_pairs()
#net = test_siamese()
embedding_train = net.predict_embedding(X_train)
plot_embedding(embedding_train, y_train)
embedding_test = net.predict_embedding(X_test)
plot_embedding(embedding_test, y_test)
plot_file = os.path.join(PLOT_PATH, 'test')
np.savetxt(plot_file + '-vec.csv', embedding_train, delimiter="\t")
labels, _ = pd.factorize(y_train)
np.savetxt(plot_file + '-meta.csv', labels.astype(int), fmt = '%d')


grouping_labels = train_data['seq_train'].values
train_metrics = metrics(embedding_train, X_train, y_train,
                        grouping_labels = grouping_labels, plot = True)
grouping_labels = test_data['seq_train'].values
test_metrics = metrics(embedding_test, X_test, y_test,
                        grouping_labels = grouping_labels, plot = True)
