#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 13:28:59 2021

@author: xgarzb@GU.GU.SE
"""
import tensorflow as tf
import tensorflow.keras.backend as K

#def triplet_loss1(y_true, dists):
#    margin = 1
#    dist_diff = tf.subtract(dists[:, 0], dists[:, 1])
#    return tf.reduce_mean(tf.maximum(dist_diff + margin, 0)) 


#def triplet_loss0(y_true, y_pred, margin = 1):

#    total_length = y_pred.shape.as_list()[-1]
#    anchor = y_pred[:,0:int(total_length/3)]
#    positive = y_pred[:,int(total_length/3):int(total_length*2/3)]
#    negative = y_pred[:,int(total_length*2/3):total_length]
#    dist_pos = tf.sqrt(tf.reduce_sum(tf.pow(anchor - positive, 2), 1, 
#                                  keepdims=True))
#    dist_neg = tf.sqrt(tf.reduce_sum(tf.pow(anchor - negative, 2), 1, 
#                                  keepdims=True))
#    return tf.reduce_mean(tf.maximum(dist_pos - dist_neg + margin, 0)) 


def triplet_loss(y_true, dist_diff):
    margin = 0.5
    return tf.reduce_mean(tf.maximum(dist_diff + margin, 0)) 


def contrastive_loss(y_true, y_pred):
    # y_true is 1 for equal labels!
    margin = 1.0

    square_pred = tf.math.square(y_pred)

    margin_square = tf.math.square(tf.math.maximum(margin - y_pred, 0))
    return tf.math.reduce_mean((1 - y_true) * square_pred + 
                               y_true * margin_square)

