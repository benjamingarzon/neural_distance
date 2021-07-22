#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 12:59:49 2021

@author: benjamin.garzon@gmail.com
"""
# consider subject group
# add derivatives
# normalize?

import logging, os, glob
import pandas as pd
import numpy as np
from random import choice
from collections import defaultdict
from datetime import date, datetime
from util import get_group, process_scores
today = date.today().isoformat()
now = datetime.now().isoformat()

class Estimator():

    def __init__(self, files_dir, target_col, ModelClass, params = None, 
                 shuffle = False, grouping_variable = None):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(f"logs/log-{today}.log", 'a', 'utf-8')
        self.logger.addHandler(handler)
        self.files_dir = files_dir
        self.Model = ModelClass
        self.target_col = target_col
        self.params = params
        self.shuffle = shuffle
        self.grouping_variable = grouping_variable
        
    def parse_files(self):
        """
        Find files and extract relevant info.

        Parameters
        ----------
        files_dir : str
            Directory with data files. 

        Returns
        -------
        None. 

        """
        file_list = glob.glob(os.path.join(self.files_dir, '*'))
        subjects = []
        self.files = {}
        self.sessions = defaultdict(list)
        labels = []
        
        for file in file_list:
            filename = os.path.basename(file)
            subject, sess, _, _, _, hemi, hemil, label = filename.split('_')
            session = int(sess[4])
            label = hemil + '_' + label.split('.')[0]
            subjects.append(subject)
            labels.append(label)
            self.files[(subject, label, session)] = file
            if session not in self.sessions[subject]:
                self.sessions[subject].append(session)
            
        self.subjects = list(set(subjects))
        self.labels = list(set(labels))
        
        print("Parsed {} files".format(len(file_list)))
        print("Found {} subjects: {}".format(len(self.subjects), self.subjects))

              
    def estimate(self, subjects = None, labels = None, selected_sessions = None, 
                 only_same_session = False):
        """
        Run through the lists and estimate the parameters. These are saved in 
        a dict (label, sess_train, sess_test, seq_class).

        Parameters
        ----------
        subjects : list
        labels : list
        sessions : list

        Returns
        -------
        None. 

        """
        #self.logger.info("Starting estimation")

        subjects = self.subjects if subjects is None else subjects
        labels = self.labels if labels is None else labels
        sessions = self.sessions
        scores = defaultdict(list) 
        for label in labels:
            for subject in subjects: # could be sampled
                subject_group = get_group(subject)
                mysessions = sessions[subject] if selected_sessions is None \
                    else [ x for x in sessions[subject] if x in selected_sessions ]
                    
                if len(mysessions) < 2: 
                    continue

                for session_train in mysessions:

                    #self.logger.info("Training:{}, {}, {}".format(subject, 
                    #                                              label, 
                    #                                              session_train))

                    #print((subject, label, session_train))
                    model = self.Model(self.target_col, 
                                       model_ref = 'exp_label-{}_sub-{}_ses-{}-{}'.format(
                                           label, 
                                           subject, 
                                           session_train, 
                                           self.params['exp_num']), 
                                       params = self.params,
                                       logger = self.logger,
                                       shuffle = self.shuffle,
                                       grouping_variable = self.grouping_variable)
                    # ad one randomly selected test session
                    session_test = choice([ x for x in mysessions if x != session_train ])
                    train_file = self.files[(subject, label, session_train)]
                    test_file = self.files[(subject, label, session_test)]
                    if only_same_session:
                        cv_score = model.fit_and_predict_file_cv(train_file)
                    if not only_same_session:
                        model.fit_file(train_file, test_file)
                    # now test in other sessions
                    for session_test in mysessions: # add type as well
                    
                            index = (label, subject_group, session_train, session_test)
                            
                            if session_test != session_train: 
                                if not only_same_session:
                                    score = model.predict_file(
                                        self.files[(subject, label, session_test)])
                                    scores[index].append(score)
                                    self.logger.info("Scores on training:{}, {}, {} \n Params:{}".format(subject, index, scores[index], self.params))

                            else:
                                if only_same_session: 
                                    scores[index].append(cv_score)
                                    self.logger.info("Scores on training:{}, {}, {} \n Params:{}".format(subject, index, scores[index], self.params))

                                #if cv_score > 10:
        #print(scores) 
        scores_df = process_scores(scores)

        mean_score = scores_df.value[scores_df.metric == 'ratio'].mean()
#        mean_score = np.mean([np.nanmean(x) for x in scores.values()])
#        self.logger.info("Finished experiment, Params:{}".format(self.params))
#        if mean_score > 1:
#            self.logger.info("Mean scores on test: {}; Params:{}".format(mean_score, self.params))
        return(scores_df, mean_score)
