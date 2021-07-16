#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 10:46:49 2021

@author: benjamin.garzon@gmail.com
"""

FILES_DIR = '/data/lv0/MotorSkill/fmriprep/analysis/roi_data'
TARGET_COL = 'true_sequence' # 'seq_type'
TRAIN_FILE = '/data/lv0/MotorSkill/fmriprep/analysis/roi_data/sub-lue1105_ses-5_roi_data_mask_rh_R_SPL.csv'
TEST_FILE = '/data/lv0/MotorSkill/fmriprep/analysis/roi_data/sub-lue1105_ses-6_roi_data_mask_rh_R_SPL.csv'
OUTPUT_DIR = '/data/lv0/MotorSkill/fmriprep/analysis/results'

TEST_LABEL = ['R_SPL']
NUM_CORES = 30

DEFAULT_SIAMESE_PARAMS = {
    'n1' : 20,
    'n2' : 20,
    'dropout' : 0,
    'activation' : 'tanh',
    'embedding_dimension' : 10,
    'loss' : 'contrastive_loss',
    'batch_size' : 64,
    'epochs' : 3
    }

PARAM_GRID = {
    'n1' : [20, 50, 100, 200],
    'n2' : [0, 50, 100, 200],
    'dropout' : [0, 0.2],
    'activation' : ['tanh', 'sigmoid', 'elu'],
    'embedding_dimension' : [5, 10, 15, 20],
    'loss' : ['binary_crossentropy', 'contrastive_loss'],
    'batch_size' : [32, 64],
    'epochs' : [50, 100]

}