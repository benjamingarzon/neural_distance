#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 10:59:30 2021

@author: benjamin.garzon@gmail.com
"""
from itertools import combinations, product
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn import manifold
import pandas as pd
from sklearn.decomposition import PCA
import random
import pickle
from itertools import product
from config import MINCORRECT, NTYPES

def get_correct(data):
        valid_types = data.groupby(['run', 'seq_type']).iscorrect.sum()
        valid_runs = (valid_types >= MINCORRECT).groupby('run').sum() == NTYPES
        valid_runs = valid_runs.index[valid_runs]
        valid = np.logical_and(data.run.isin(valid_runs),
                               data.iscorrect)
        data = data[data.valid]
        nvalid = np.sum(valid)
        ntrials = len(valid)
        nruns = len(valid_runs)

        return data, ntrials, nvalid, nruns

def balance_labels(df, target_col):
     
    dfx = df[['iscorrect', target_col, 'run']].loc[df.iscorrect]
    mintrials = dfx.groupby([target_col, 'run']).count().groupby('run').min()
    runs = df.run.unique()
    labels = df[target_col].unique()
    index_list = []
    for run, label in product(runs, labels): 
        indices = dfx.loc[np.logical_and(dfx.run == run, dfx[target_col] == label)].index.tolist()
        index_list.extend(random.sample(indices,  k = int(mintrials.loc[run])))
    
    df_balanced = df.loc[sorted(index_list)]
    #print(df_balanced[['iscorrect', target_col, 'run']].groupby([target_col, 'run']).count().sort_values('run'))
    
    return df_balanced

def create_pairs(X, labels, sample = False):
    """
    Find files and extract relevant info.

    Parameters
    ----------
    X (pandas df): 

    labels ():
        
    Returns
    -------
    index_pairs (list) :
    label_pairs (list) :

    """
    print("Creating pairs")
    index_pairs = list(combinations(range(X.shape[0]), 2))
    label_pairs = np.array([1 * (labels[i] == labels[j]) for i, j in index_pairs ])
    if sample:
        # not implemented, select only a sample of them
        pass
    
    X_pairs = [(X[i, :], X[j, :]) for i, j in index_pairs ]
    X_1, X_2 = zip(*X_pairs)
    X_1 = np.array(X_1)
    X_2 = np.array(X_2)
    X_pairs = [X_1, X_2]
    return(index_pairs, X_pairs, label_pairs)

def euclidean_distance(z):
	x, y = z
	sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
	return K.sqrt(K.maximum(sum_square, K.epsilon()))

def contrastive_loss(y_true, y_pred):
    y_true=tf.dtypes.cast(y_true, tf.float64)
    y_pred=tf.dtypes.cast(y_pred, tf.float64)
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def loop_params(params_grid, sample = None):
    params = list(product(*params_grid.values()))
    params_dicts = [dict(zip(params_grid.keys(), x)) for x in params]
    print("Parameter combinations: {}".format(len(params_dicts)))
    if sample is not None:
        params_dicts = random.sample(params_dicts, k = min(sample, len(params_dicts)))
    # add ref number
    for i, params in enumerate(params_dicts):
        params['exp_num'] = '%0.5d'%(i)        
    return(params_dicts)

def distance_ratio(embeddings, index_test, label_test, plot = False):
    same, different = [] , []
    for index, (i, j) in enumerate(index_test):
        dist = euclidean_distance([embeddings[[i], :], embeddings[[j], :]]).numpy()
    
        if label_test[index] == 1: #same class
            same.append(dist)
        else:
            different.append(dist)
    if plot:
        plt.figure()
        plt.hist(np.array(same).flatten(), 30, color = 'r', alpha = 0.5)
        plt.hist(np.array(different).flatten(), 30, color = 'b', alpha = 0.5)
        plt.show()
    same = np.median(same)
    different = np.median(different)
    ratio = (different/same - 1)*100
    print('Distance ratio:', same, different, ratio)
    return(ratio)
    
def plot_training(H, plot_file, test = True):
    # construct a plot that plots and saves the training history
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H.history["loss"], label="train_loss")
    if test:
        plt.plot(H.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch #")
    plt.title("Loss")   
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    if plot_file:
        plt.savefig(plot_file + '-loss.png')
    else:
        plt.show()
    plt.figure()
    plt.plot(H.history["accuracy"], label="train_acc")
    if test:
        plt.plot(H.history["val_accuracy"], label="val_acc")
    plt.xlabel("Epoch #")
    plt.title("Accuracy")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.axhline(y=0.25, color='k', linestyle='-')
    plt.legend(loc="lower left")
    if plot_file:
        plt.savefig(plot_file + '-acc.png')
    else:
        plt.show()
    
def plot_embedding(embedding, y, plot_file):
    n_components = 2
    n_neighbors = 10
    methods = {}
#    methods['t-SNE'] = manifold.TSNE(n_components=n_components, init='pca',
#                                 random_state=0)
    methods['PCA'] = PCA(n_components=n_components)
    labels = pd.factorize(y)[0]
#    markers = {1:'o', 2:'v', 3:'s', 4:'P', 5: '*'}
    # Plot results
    for i, (labelx, method) in enumerate(methods.items()):
        Y = method.fit_transform(embedding)
        plt.figure()
        plt.scatter(Y[:, 0], Y[:, 1], 
                       c = labels*10,
                       s = 30,
                       cmap = plt.cm.Spectral)
        plt.title(labelx)
    if plot_file:
        plt.savefig(plot_file + '-embed.png')
    else:
        plt.show()
def plot_results(results_file):
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)   
    param_names = results['params_list'][0].keys()
    plt.style.use("ggplot")

    for param_x in param_names:
#        for param_y in param_names:
        x = [v[param_x] for v in results['params_list']]
        x = np.array( x )
        if len(set(x)) == 1:
            continue
#        if param_x != param_y:
#        y = [v[param_y] for v in results['params_list']]
#        y = np.array(y)
        z = np.array(results['score_list'])
        plt.scatter(x, z) #, c = pd.factorize(y)[0])
        plt.xlabel(param_x)
        plt.ylabel('Score')
        plt.legend(loc="lower left")
        title = '%s.png'%(param_x) #, param_y)
        plt.title(title)
        plt.show() #savefig(title + '.png')
