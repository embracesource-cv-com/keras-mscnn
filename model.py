# -*- coding:utf-8 -*-
"""
   File Name:     model.py
   Description:   模型定义
   Author:        steven.yi
   date:          2019/04/25
"""
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model


def msb_module(kernel_list, numbers):
    """
    Multi-Scale Blob module

    kernel_list: list of int, e.g. [3,5,7] means 3 filters with
                 size(3x3,5x5,7x7) in one MSB module

    numbers: int, how much numbers of MSB module in this layer

    input shape: (samples, height, width, channels)
    output shape: (samples, height, width, len(kernel_list) * numbers)
    """

    def f(x):
        outputs = []
        for kernel in kernel_list:
            outputs.append(Conv2D(numbers, kernel, activation='relu', padding='same')(x))
        outputs = Concatenate(axis=-1)(outputs)
        outputs = BatchNormalization()(outputs)
        outputs = Activation('relu')(outputs)
        return outputs

    return f
