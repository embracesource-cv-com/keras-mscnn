# -*- coding:utf-8 -*-
"""
   File Name:     metrics.py
   Description:   评估标准
   Author:        steven.yi
   date:          2019/05/05
"""
import keras.backend as K


def mae(y_true, y_pred):
    return K.abs(K.sum(y_true) - K.sum(y_pred))


def mse(y_true, y_pred):
    return (K.sum(y_true) - K.sum(y_pred)) * (K.sum(y_true) - K.sum(y_pred))