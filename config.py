#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 10:46:49 2021

@author: benjamin.garzon@gmail.com
"""
import os 
FILES_DIR = '/data/lv0/MotorSkill/fmriprep/analysis/roi_data'
TARGET_COL = 'true_sequence' # 'seq_type'
TRAIN_FILE = '/data/lv0/MotorSkill/fmriprep/analysis/roi_data/sub-lue1107_ses-4_roi_data_mask_rh_R_SPL.csv'
TEST_FILE = '/data/lv0/MotorSkill/fmriprep/analysis/roi_data/sub-lue1107_ses-5_roi_data_mask_rh_R_SPL.csv'
OUTPUT_DIR = '/data/lv0/MotorSkill/fmriprep/analysis/results'

SCORES_PATH = os.path.join(OUTPUT_DIR, "scores.csv")
MODEL_PATH = os.path.join(OUTPUT_DIR, "models")
PLOT_PATH = os.path.join(OUTPUT_DIR, "plots")
SHUFFLE = False
MINCORRECT = 3
NTYPES = 4       

PARAMETER_TUNING = True 
NUM_CORES = 10
BLOCK_SIZE = NUM_CORES

TEST_LABEL = ['R_SPL']

DEFAULT_TRIPLET_PARAMS = {
    'n1' : 20,
    'n2' : 0,
    'dropout' : 0,
    'activation' : 'relu',
    'embedding_dimension' : 10,
    'batch_size' : 128,
    'epochs' : 5,
    'exp_num' : '0000',
    'learning_rate': 0.001,
    'modelclass': 'TripletNet'
    }

DEFAULT_SIAMESE_PARAMS = {
    'n1' : 30,
    'n2' : 0,
    'dropout' : 0.2,
    'activation' : 'relu',
    'embedding_dimension' : 10,
    'loss' : 'contrastive_loss', #
    'batch_size' : 64,
    'epochs' : 30,
    'exp_num' : '0000',
    'learning_rate': 0.001,
    'modelclass': 'SiameseNet_cont'
    }

PARAM_GRID = {
    'n1' : [10, 50, 100],
    'n2' : [0, 20],
    'dropout' : [0, 0.2],
    'activation' : ['relu', 'sigmoid', 'tanh'],# 'elu'],
    'embedding_dimension' : [5, 10, 15],
    'batch_size' : [64], #32
    'learning_rate': [0.01],
    'modelclass': ['SiameseNet_cont', 'SiameseNet_bin', 'TripletNet']
#    'epochs' : [50], 
#    'loss' : ['contrastive_loss', 'binary_crossentropy'], 

}