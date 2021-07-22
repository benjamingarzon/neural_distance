#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 10:46:49 2021

@author: benjamin.garzon@gmail.com
"""
import os 
FILES_DIR = '/data/lv0/MotorSkill/fmriprep/analysis/roi_data'
TARGET_COL = 'true_sequence' # 'seq_type'
TRAIN_FILE = '/data/lv0/MotorSkill/fmriprep/analysis/roi_data/sub-lue1107_ses-5_roi_data_mask_rh_R_SPL.csv'
TEST_FILE = '/data/lv0/MotorSkill/fmriprep/analysis/roi_data/sub-lue1107_ses-6_roi_data_mask_rh_R_SPL.csv'
OUTPUT_DIR = '/data/lv0/MotorSkill/fmriprep/analysis/results'

SCORES_PATH = os.path.join(OUTPUT_DIR, "scores.csv")
MODEL_PATH = os.path.join(OUTPUT_DIR, "models")
PLOT_PATH = os.path.join(OUTPUT_DIR, "plots")


MINCORRECT = 3
NTYPES = 4        

TEST_LABEL = ['R_SPL']
NUM_CORES = 25

DEFAULT_TRIPLET_PARAMS = {
    'n1' : 10,
    'n2' : 10,
    'dropout' : 0,
    'activation' : 'relu',
    'embedding_dimension' : 5,
    'batch_size' : 64,
    'epochs' : 100,
    'exp_num' : '0000',
    'learning_rate': 0.001
    }

DEFAULT_SIAMESE_PARAMS = {
    'n1' : 30,
    'n2' : 0,
    'dropout' : 0.2,
    'activation' : 'relu',
    'embedding_dimension' : 10,
    'loss' : 'binary_crossentropy',  #'contrastive_loss', #
    'batch_size' : 128,
    'epochs' : 30,
    'exp_num' : '0000',
    'learning_rate': 0.001
    }

PARAM_GRID = {
    'n1' : [10, 50, 100],
    'n2' : [0, 20, 50],
    'dropout' : [0, 0.2],
    'activation' : ['relu', 'sigmoid'],# 'elu'],
    'embedding_dimension' : [5, 10, 15],
    'loss' : ['contrastive_loss', 'binary_crossentropy'], 
    'batch_size' : [64], #32
    'epochs' : [100], 
    'learning_rate': [0.01, 1e-4]

}