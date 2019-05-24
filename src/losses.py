# -*- coding: utf-8 -*-
"""
   File Name：     losses.py
   Description :   损失函数
   Author :       mick.yi
   Date：          2019/5/24
"""
import tensorflow as tf


def l2(y_true, y_pred):
    """
    l2损失
    :param y_true: [batch_size,H,W,1]
    :param y_pred: [batch_size,H,W,1]
    :return:
    """
    return tf.reduce_sum((y_true - y_pred) ** 2)
