#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 14:14:41 2021

@author: benjamin.garzon@gmail.com
"""
# to params=
import os, logging
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, Flatten
import tensorflow as tf
#Conv3D,GlobalAveragePooling3D, MaxPooling3D, 
from util import euclidean_distance, contrastive_loss, plot_training, \
    create_pairs
from config import DEFAULT_SIAMESE_PARAMS
import pandas as pd
import numpy as np
tf.keras.backend.clear_session()

BASE_OUTPUT = "output"
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])

class SiameseNet():
    def __init__(self, target_col, params = None):
        self.target_col = target_col
        if params is None:
            self.params = DEFAULT_SIAMESE_PARAMS
        else:
            self.params = params
        
    def build(self, input_shape):
	# second set of CONV => RELU => POOL => DROPOUT layers
	#x = Conv2D(64, (2, 2), padding="same", activation="relu")(x)
	#x = MaxPooling2D(pool_size=2)(x)
	#x = Dropout(0.3)(x)

        # definition 1
        inputs = Input(input_shape)
#        x = Flatten()(inputs) #??
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
        
        self.sister.summary()

        X_1 = Input(shape=input_shape)
        X_2 = Input(shape=input_shape)

        emb_1 = self.sister(X_1)
        emb_2 = self.sister(X_2)

        if self.params['loss'] == 'binary_crossentropy': 
            dist = Lambda(euclidean_distance)([emb_1, emb_2])
            outputs = Dense(1, activation="sigmoid")(dist)
            self.model = Model([X_1, X_2], outputs)
            self.model.compile(loss="binary_crossentropy", optimizer="adam",
                          metrics=["accuracy"])
        else: 
            outputs = Lambda(euclidean_distance)([emb_1, emb_2])
            self.model = Model([X_1, X_2], outputs)
            self.model.compile(loss=contrastive_loss, optimizer="adam",
                          metrics=["accuracy"])

        self.model.summary()

    def fit(self, X_pairs_train, label_train, X_pairs_test = None, 
            label_test = None, plot = False):
        
        if label_test is not None:
            self.history = self.model.fit(
            X_pairs_train, label_train,
            	validation_data=(X_pairs_test, label_test),
            	batch_size = self.params['batch_size'], 
            	epochs = self.params['epochs'])
        else:
            self.history = self.model.fit(
            X_pairs_train, label_train,
            	batch_size = self.params['batch_size'], 
            	epochs = self.params['epochs'])
        #print("[INFO] saving siamese model...")
        #self.model.save(MODEL_PATH)
        # plot the training history
        #print("[INFO] plotting training history...")
        if plot:
            plot_training(self.history, PLOT_PATH, 
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
        
        train_data = pd.read_csv(train_file)
        self.feature_cols = [ x for x in train_data.columns if 'feature' in x ]
        y_train = train_data[self.target_col].values
        X_train = train_data[self.feature_cols].values
        self.trained_labels = np.unique(y_train)

        index_train, X_pairs_train, label_train = create_pairs(X_train, y_train)
 
        self.build(input_shape = (X_train.shape[1]) )
        
        if test_file is not None:
            test_data = pd.read_csv(test_file)
            y_test = test_data[self.target_col]
            # remove labels that were not trained
            X_test = test_data[self.feature_cols]
            X_test = X_test.loc[y_test.isin(self.trained_labels), :].values
            y_test = y_test.loc[y_test.isin(self.trained_labels)].values
            
            index_test, X_pairs_test, label_test = create_pairs(X_test, y_test)
        else:
            X_pairs_test, label_test = None, None
        
        self.fit(X_pairs_train, label_train, X_pairs_test, label_test)
        #fitted_model = self.model.fit(X, y)
        #return fitted_model

        # train the model
    
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
        test_data = pd.read_csv(test_file)
        y_test = test_data[self.target_col]
        # remove labels that were not trained
        X_test = test_data[self.feature_cols]
        X_test = X_test.loc[y_test.isin(self.trained_labels), :].values
        y_test = y_test.loc[y_test.isin(self.trained_labels)].values

        index_test, X_pairs_test, label_test = create_pairs(X_test, y_test)

        embeddings = self.predict_embedding(X_test)
        same, different = [] , []
        for index, (i, j) in enumerate(index_test):
            dist = euclidean_distance([embeddings[[i], :], embeddings[[j], :]]).numpy()
        
            if label_test[index] == 1: #same class
                same.append(dist)
            else:
                different.append(dist)
            score = np.mean(np.array(different)) / np.mean(np.array(same))
        return(score)
        
