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
    if sample is not None:
        params_dicts = random.sample(params_dicts, k = sample)
        
    return(params_dicts)
        
    
def plot_training(H, plotPath, test = True):
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
    plt.figure()
    plt.plot(H.history["accuracy"], label="train_acc")
    if test:
        plt.plot(H.history["val_acc"], label="val_acc")
    plt.xlabel("Epoch #")
    plt.title("Accuracy")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.axhline(y=0.25, color='r', linestyle='-')
    plt.legend(loc="lower left")
    plt.show()
    #plt.savefig(plotPath)
    
def plot_embedding(embedding, y):
    n_components = 2
    n_neighbors = 10
    methods = {}
    methods['t-SNE'] = manifold.TSNE(n_components=n_components, init='pca',
                                 random_state=0)
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

    plt.show()