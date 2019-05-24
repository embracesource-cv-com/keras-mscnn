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
from keras.regularizers import l2


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
            outputs.append(Conv2D(numbers, kernel, padding='same',
                                  kernel_regularizer=l2(5e-4))(x))
        outputs = Concatenate(axis=-1)(outputs)
        outputs = BatchNormalization()(outputs)
        outputs = Activation('relu')(outputs)
        return outputs

    return f


def MSCNN(input_shape):
    inputs = Input(shape=input_shape)

    # conv
    x = Conv2D(64, (9, 9), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # MSB conv
    x = msb_module([9, 7, 5, 3], 16)(x)
    # down sample
    x = MaxPooling2D(2)(x)
    # MSB conv
    x = msb_module([9, 7, 5, 3], 32)(x)
    x = msb_module([9, 7, 5, 3], 32)(x)
    # down sample
    x = MaxPooling2D(2)(x)
    # MSB conv
    x = msb_module([7, 5, 3], 64)(x)
    x = msb_module([7, 5, 3], 64)(x)
    # density map regression
    # x = Conv2D(1000, (1, 1), padding='same', kernel_regularizer=l2(5e-4))(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    x = Conv2D(1, (1, 1), activation='sigmoid')(x)
    density_map = Activation('relu')(x)
    model = Model(inputs=inputs, outputs=density_map)
    return model
