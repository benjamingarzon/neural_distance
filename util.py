#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 10:59:30 2021

@author: benjamin.garzon@gmail.com
"""
from itertools import combinations, product
import tensorflow.keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold
import pandas as pd
from sklearn.decomposition import PCA
import random
import pickle
from itertools import product
from config import MINCORRECT, NTYPES

def get_group(subject):

    if any([ 'lue%d1'%x in subject for x in range(1, 6)]): 
        return 'Intervention'
    else:
        return 'Control'        
    
    
def get_correct(data):
        data.dropna(subset = ['iscorrect'], inplace = True)
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
    index_pairs = list(combinations(range(X.shape[0]), 2))
    random.shuffle(index_pairs)
    label_pairs = np.array([1 * (labels[i] == labels[j]) for i, j in index_pairs ])
    if sample:
        # not implemented, select only a sample of them
        pass
#    if False: ### simluate some data easy to learn
#        ff, _ = pd.factorize(labels)
#        X_pairs = [(X[i, :]/500+ ff[i], X[j, :]/500 + ff[j]) for i, j in index_pairs ]
#        X_1, X_2 = zip(*X_pairs)
#        X_1 = np.array(X_1)
#        X_2 = np.array(X_2)
#        dd = np.sqrt(np.sum((X_1 - X_2)**2, axis = 1))
#        plt.figure()
#        plt.scatter(label_pairs, dd)
#        plt.show()
#    else:
    X_pairs = [(X[i, :], X[j, :]) for i, j in index_pairs ]

    X_1, X_2 = zip(*X_pairs)
    X_1 = np.array(X_1)
    X_2 = np.array(X_2)
    X_pairs = [X_1, X_2]
    print("Created %d pairs"%len(index_pairs))
#    print(*label_pairs, sep = ' ')
    return(index_pairs, X_pairs, label_pairs)

def create_triplets(X, labels, sample = None):
    #index_pairs, X_pairs, label_pairs = create_pairs(X, labels)
    indices = range(X.shape[0])

    index_prod = product(indices, indices, indices)
    
    index_triplets = []
 
    for (i, j, k) in index_prod:
        if labels[i] == labels[j] and i!=j and labels[i]!= labels[k]:
            index_triplets.append((i, j, k))

    random.shuffle(index_triplets)
    
    if sample is not None and sample < len(index_triplets):
        index_triplets = random.sample(index_triplets, k = sample)
        
    X_triplets = []
    for (i, j, k) in index_triplets:
        X_triplets.append((X[i, :], X[j, :], X[k, :]))

    X_1, X_2, X_3 = zip(*X_triplets)
    X_1 = np.array(X_1)
    X_2 = np.array(X_2)
    X_3 = np.array(X_3)
    X_triplets = [X_1, X_2, X_3]

    print("Created %d triplets"%len(index_triplets))

    return(index_triplets, X_triplets, np.zeros(len(index_triplets)))
    
def euclidean_distance(z):
    """
    Divided by num dimension to make it comparable
    """
    x, y = z
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)/tf.dtypes.cast(x.shape[-1], tf.float32)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def squeeze_dist_diff(x):
    dist_pos = euclidean_distance([x[0], x[1]])
    dist_neg = euclidean_distance([x[0], x[2]])
    return(tf.math.sigmoid(dist_pos - dist_neg))

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

def process_scores(scores):
    
    # turn into
    data = []
    for score_key, score_value in scores.items():
        for x in score_value:
            if type(x) == dict:
                for metrics_key, metrics_value in x.items():
                    for train_key, train_value in metrics_value.items():
                        data.append(score_key + (metrics_key, train_key, train_value))
    score_df = pd.DataFrame(data, columns = ['label', 'group' ,'session_train', 
                                           'session_test', 'metric', 'seq_train', 'value'])
    return(score_df)
    
def accuracy(embeddings, index_triplets):

    els = np.zeros(len(index_triplets))
    
    for index, (i, j, k) in enumerate(index_triplets):
        dist_pos = euclidean_distance([embeddings[[i], :], embeddings[[j], :]]).numpy()
        dist_neg = euclidean_distance([embeddings[[i], :], embeddings[[k], :]]).numpy()
        els[index] = dist_pos - dist_neg < 0

    acc = np.mean(els)
    print('Accuracy:', acc)
    return acc 

def distance_ratio(embeddings, index_test, label_test, plot = False):
    same, different = [] , []
    for index, (i, j) in enumerate(index_test):
        dist = euclidean_distance([embeddings[[i], :], embeddings[[j], :]]).numpy()
    
        if label_test[index] == 1: #same class
            same.append(dist**2)
        else:
            different.append(dist**2)
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

def metrics(embeddings, X_test, y_test, grouping_labels, plot = False, 
            sample_triplets = 10000):
    
    n_labels = len(np.unique(y_test)) 

    index_pairs, X_pairs, label_pairs = create_pairs(X_test, y_test)
    index_triplets, X_triplets, label_triplets = create_triplets(X_test, y_test, 
                                                                 sample = sample_triplets)

    if grouping_labels is None:
        ratio = distance_ratio(embeddings, index_pairs, label_pairs)
        acc = accuracy(embeddings, index_triplets)*n_labels/4
        metrics = {'acc': acc, 'ratio': ratio}
        return metrics
    else:
        grouping_pairs = []
        for (i, j) in index_pairs:
             if grouping_labels[i] == grouping_labels[j]:
                 grouping_pairs.append(grouping_labels[i])
             else:
                 grouping_pairs.append(
                     ('-').join(sorted([grouping_labels[i], grouping_labels[j]])))

        grouping_triplets = []
        for (i, j, k) in index_triplets:
            if grouping_labels[i] == grouping_labels[k]:
                grouping_triplets.append(grouping_labels[i])
            else:
                grouping_triplets.append(
                    ('-').join(sorted([grouping_labels[i], grouping_labels[k]])))

        grouping_pairs = np.array(grouping_pairs)
        grouping_triplets = np.array(grouping_triplets)
        grouping_classes = np.unique(grouping_pairs)
        ratios = {}
        accs = {}
        
        for g in grouping_classes:
            print(g)
            #embeddings_group =  embeddings[grouping_labels == g, :]
            myslice = [i for i, x in enumerate(grouping_pairs) if g == x]
            index_group = [index_pairs[i] for i in myslice]
            label_group = np.array([label_pairs[i] for i in myslice])
            ratios[g] = distance_ratio(embeddings, index_group, label_group)

            myslice = [i for i, x in enumerate(grouping_triplets) if g == x]
 
            index_group = [index_triplets[i] for i in myslice]
            accs[g] = accuracy(embeddings, index_group)*n_labels/4

        metrics = {'acc': accs, 'ratio': ratios}
        return metrics

    
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
    
def plot_embedding(embedding, y, plot_file = None):
    n_components = 3
    n_neighbors = 10
    methods = {}
#    methods['t-SNE'] = manifold.TSNE(n_components=n_components, init='pca',
#                                 random_state=0)
    methods['PCA'] = PCA(n_components=n_components)
    labels = pd.factorize(y)[0]+1
    markers = {1:'o', 2:'v', 3:'s', 4:'P', 0: '*'}
    colors = {1:'g', 2:'k', 3:'b', 4:'r', 0: 'y'}
    # Plot results
    for i, (labelx, method) in enumerate(methods.items()):
        Y = method.fit_transform(embedding)
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for label in np.unique(labels):
            ax.scatter(Y[labels == label, 0], 
                       Y[labels == label, 1], 
                       Y[labels == label, 2], 
                       c = colors[label],
                       s = 30,  
                       marker = markers[label])
#                           cmap = plt.cm.Spectral)
        plt.title(labelx)
    if plot_file is not None:
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


def plot_scores(scores_df):
    import seaborn as sns
#    plt.figure(figsize = (30, 30))
# labels, trained / untrained / same /different / session_trained / session_/test
    df = scores_df.loc[scores_df.metric == 'ratio']
    g = sns.FacetGrid(df,  col="session_train", row = "seq_train") #,col="group", c = "seq_train"
    g.map(sns.scatterplot, "session_test", "value")
    