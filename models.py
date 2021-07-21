#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 14:14:41 2021

@author: benjamin.garzon@gmail.com
"""
# to params=
import os, logging
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, Flatten
from tensorflow.keras.callbacks import TensorBoard
#Conv3D,GlobalAveragePooling3D, MaxPooling3D, 
from util import euclidean_distance, contrastive_loss, plot_training, \
    create_pairs, distance_ratio, plot_embedding, balance_labels, get_correct
from config import DEFAULT_SIAMESE_PARAMS, PLOT_PATH, MODEL_PATH
import pandas as pd
import numpy as np
from sklearn.model_selection import PredefinedSplit

tf.keras.backend.clear_session()

class Net():

    def __init__(self, target_col, 
                 model_ref = None, 
                 params = None, 
                 logger = None, 
                 shuffle = False):
        self.target_col = target_col
        self.params = params
        self.model_ref = model_ref
        self.acc_function = distance_ratio
        self.logger = logger
        self.shuffle = shuffle # shuffle data to have a baseline
        self.trained = False
        
    def log(self, message):
        if self.logger:
            self.logger.info(message)
    

    def fit(self, X_pairs_train, label_train, X_pairs_test = None, 
            label_test = None, plot = True, save_model = True):
        
        tb_path = os.path.join(MODEL_PATH, self.model_ref, 'tb')
        save_path = os.path.join(MODEL_PATH, self.model_ref, 'save')
    
        tensorboard_callback = TensorBoard(log_dir = tb_path, histogram_freq=1)

        if label_test is not None:
            self.history = self.model.fit(
            X_pairs_train, label_train,
            	validation_data=(X_pairs_test, label_test),
            	batch_size = self.params['batch_size'], 
            	epochs = self.params['epochs'], 
            callbacks = [tensorboard_callback])
        else:
            self.history = self.model.fit(
            X_pairs_train, label_train,
            	batch_size = self.params['batch_size'], 
            	epochs = self.params['epochs'], 
            callbacks = [tensorboard_callback])
        #print("[INFO] saving siamese model...")
        self.model.save(save_path)
        if plot:
            plot_file = os.path.join(PLOT_PATH, self.model_ref) if self.model_ref else None
            plot_training(self.history, 
                          plot_file, 
                          test = label_test is not None)

    def fit_file(self, train_file, test_file = None):         
        """
        Fit model to data.

        Parameters
        ----------

        Returns
        -------
        None. 

        """
        train_data, ntrials, nvalid, nruns = get_correct(pd.read_csv(train_file))
        self.feature_cols = [ x for x in train_data.columns if 'feature' in x ]

        runs = np.unique(train_data.run)
        if nruns < 4:
            self.log('Skipping, less than 4 runs: %s'%train_file)
            return None
        else: 
            self.log('%s has %d valid runs'%(train_file, len(runs)))
        train_data = balance_labels(train_data, self.target_col)
        
        y_train = train_data[self.target_col].values
        X_train = train_data[self.feature_cols].values/self.params['scaling']
        self.trained_labels = np.unique(y_train)

        if self.shuffle:
            self.log('shuffling data')
            for run in runs:
                y_train[train_data.run == run] = \
                    np.random.permutation(y_train[train_data.run == run])    

        index_train, X_pairs_train, label_train = create_pairs(X_train, y_train)
 
        self.build(input_shape = (X_train.shape[1]) )
        
        if test_file is not None:

            test_data, *_ = get_correct(pd.read_csv(test_file))
            # remove labels that were not trained and balance
            test_data = test_data.loc[test_data[self.target_col].isin(self.trained_labels)]
            test_data = balance_labels(test_data, self.target_col)
            X_test = test_data[self.feature_cols].values/self.params['scaling']
            y_test = test_data[self.target_col].values

            index_test, X_pairs_test, label_test = create_pairs(X_test, y_test)
        else:
            X_pairs_test, label_test = None, None
        
        self.fit(X_pairs_train, label_train, X_pairs_test, label_test)
        self.trained = True

    def fit_and_predict_file_cv(self, data_file):         
        """
        Fit model to data, crossvalidating within run.

        Parameters
        ----------

        Returns
        -------
        None. 

        """

        data, ntrials, nvalid, nruns = get_correct(pd.read_csv(data_file))
        self.feature_cols = [ x for x in data.columns if 'feature' in x ]

        print(ntrials, nvalid, nruns)
        if nruns < 4:
            self.log('Skipping, less than 4 runs: %s'%data_file)
            return np.nan
        else: 
            self.log('%s has %d valid runs'%(data_file, len(np.unique(data.run))))
        data = balance_labels(data, self.target_col)
        X = data[self.feature_cols].values/ self.params['scaling']
        y = data[self.target_col].values

        folds = PredefinedSplit(data.run)

        self.build(input_shape = (X.shape[1]))
        
        scores = []
        for i, (train_index, test_index) in enumerate(folds.split()):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            if self.shuffle:
                self.log('shuffling data')
                y_train = np.random.permutation(y_train)    
                
            index_train, X_pairs_train, label_train = create_pairs(X_train, y_train)
            index_test, X_pairs_test, label_test = create_pairs(X_test, y_test)
 
            self.fit(X_pairs_train, label_train, X_pairs_test, label_test)
            embeddings = self.predict_embedding(X_test)
            score = self.acc_function(embeddings, index_test, label_test)
            scores.append(score)
            plot_file = os.path.join(PLOT_PATH,  '%s-split%i'%(self.model_ref, i)) if self.model_ref else None
            plot_embedding(embeddings, 
                       y_test,
                       plot_file = plot_file)
            
        cv_score = np.mean(scores)
        return(cv_score)
        
    def predict_embedding(self, X):
        return self.sister.predict(X)
        
    def predict_file(self, test_file):
        """
        Predict from new data.

        Parameters
        ----------

        Returns
        -------
        None. 

        """

        if not self.trained:
            return np.nan
        test_data, *_ = get_correct(pd.read_csv(test_file))

        # remove labels that were not trained and balance
        test_data = test_data.loc[test_data[self.target_col].isin(self.trained_labels)]
        test_data = balance_labels(test_data, self.target_col)
        X_test = test_data[self.feature_cols].values/ self.params['scaling']
        y_test = test_data[self.target_col].values


        index_test, X_pairs_test, label_test = create_pairs(X_test, y_test)

        embeddings = self.predict_embedding(X_test)

        plot_file = os.path.join(PLOT_PATH,  '%s'%self.model_ref) if self.model_ref else None
        plot_embedding(embeddings, 
                   y_test,
                   plot_file = plot_file)

        score = self.acc_function(embeddings, index_test, label_test)

        return(score)        

class SiameseNet(Net):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        if self.params is None:
            self.params = DEFAULT_SIAMESE_PARAMS
        
    def build(self, input_shape):
	# second set of CONV => RELU => POOL => DROPOUT layers
	#x = Conv2D(64, (2, 2), padding="same", activation="relu")(x)
	#x = MaxPooling2D(pool_size=2)(x)
	#x = Dropout(0.3)(x)
        
        # definition 1
        inputs = Input(input_shape)
#        x = Flatten()(inputs) #??
#        x = Lambda(scale)(inputs)
        x = Dense(self.params['n1'], activation = self.params['activation'])(inputs)
        if self.params['dropout'] > 0:
            x = Dropout(self.params['dropout'])(x)
        if self.params['n2'] > 0:
            x = Dense(self.params['n2'], activation = self.params['activation'])(x)

            if self.params['dropout'] > 0:
                x = Dropout(self.params['dropout'])(x)

        outputs = Dense(self.params['embedding_dimension'])(x)
        # create two equal instances
        self.sister = Model(inputs, outputs)
        
        # definition 2
        #layers = [Dense(50, activation = 'relu'),
        #          Dense(embedding_dimension)]
        #sister = Sequential(layers)
        
        #self.sister.summary()

        X_1 = Input(shape=input_shape)
        X_2 = Input(shape=input_shape)

        emb_1 = self.sister(X_1)
        emb_2 = self.sister(X_2)

        if self.params['loss'] == 'binary_crossentropy': 
            dist = Lambda(euclidean_distance)([emb_1, emb_2])
            outputs = Dense(1, activation="sigmoid")(dist)
            self.model = Model([X_1, X_2], outputs)
            self.model.compile(loss="binary_crossentropy", 
                               #optimizer = "adam",
                               optimizer=Adam(learning_rate = self.params['learning_rate']),
                          metrics=["accuracy"])
        else: 
            outputs = Lambda(euclidean_distance)([emb_1, emb_2])
            self.model = Model([X_1, X_2], outputs)
            self.model.compile(loss=contrastive_loss, 
                               #optimizer = "adam",
                               optimizer=Adam(learning_rate = self.params['learning_rate']),
                          metrics=["accuracy"])

        #self.model.summary()

    
    
