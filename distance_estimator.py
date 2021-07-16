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
from collections import defaultdict
from datetime import date
today = date.today().isoformat()

class Estimator():

    def __init__(self, files_dir, target_col, ModelClass, params = None):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(f"logs/log-{today}.log", 'a', 'utf-8')
        self.logger.addHandler(handler)
        self.files_dir = files_dir
        self.Model = ModelClass
        self.target_col = target_col
        self.params = params
    
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

              
    def estimate(self, subjects = None, labels = None, selected_sessions = None):
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
                mysessions = sessions[subject] if selected_sessions is None \
                    else [ x for x in sessions[subject] if x in selected_sessions ]

                for session_train in mysessions:

                    #self.logger.info("Training:{}, {}, {}".format(subject, 
                    #                                              label, 
                    #                                              session_train))

                    print((subject, label, session_train))
                    model = self.Model(self.target_col, self.params)
                    model.fit_file(self.files[(subject, label, session_train)])
                    
                    # now test in other sessions
                    for session_test in mysessions: # add type as well
                        if session_test != session_train: 
                            index = (label, session_train, session_test)
                            score = model.predict_file(
                                self.files[(subject, label, session_test)])
                            scores[index].append(score)
                            #self.logger.info("Scores:{} ::: {}".format(index, score))

        mean_score = np.mean([np.mean(x) for x in scores.values()])
        self.logger.info("Params: {}".format(self.params))
        self.logger.info("Mean_score: {} ".format(mean_score))
        return(scores, mean_score)