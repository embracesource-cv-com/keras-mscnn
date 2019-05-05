# -*- coding:utf-8 -*-
"""
   File Name:     config.py
   Description:   模型定义
   Author:        steven.yi
   date:          2019/05/05
"""


class Config(object):
    # 模型存放目录
    MODEL_DIR = './trained_models/'

    # 训练集目录
    TRAIN_PATH = './data/formatted_trainval_{0}/shanghaitech_part_{0}_patches_9/train'
    # 训练集Ground-Truth目录
    TRAIN_GT_PATH = './data/formatted_trainval_{0}/shanghaitech_part_{0}_patches_9/train_den'

    # 验证集目录
    VAL_PATH = './data/formatted_trainval_{0}/shanghaitech_part_{0}_patches_9/val'
    # 验证集Ground_Truth目录
    VAL_GT_PATH = './data/formatted_trainval_{0}/shanghaitech_part_{0}_patches_9/val_den'

    # 测试集目录
    TEST_PATH = './data/original/shanghaitech/part_{}_final/test_data/images/'
    # 测试集Ground_Truth目录
    TEST_GT_PATH = './data/original/shanghaitech/part_{}_final/test_data/ground_truth_csv/'

    # 测试集Ground_Truth heatmap目录
    HM_GT_PATH = './heatmaps_gt'

    EPOCHS = 200
    TRAIN_BATCH_SIZE = 1
    VAL_BATCH_SIZE = 1


class LocalConfig(object):
    # 模型存放目录
    MODEL_DIR = './trained_models/'

    # 训练集目录
    TRAIN_PATH = '/opt/dataset/crowd_counting/shanghaitech/formatted_trainval_{0}/shanghaitech_part_{0}_patches_9/train'
    # 训练集Ground-Truth目录
    TRAIN_GT_PATH = '/opt/dataset/crowd_counting/shanghaitech/formatted_trainval_{0}/shanghaitech_part_{0}_patches_9/train_den'

    # 验证集目录
    VAL_PATH = '/opt/dataset/crowd_counting/shanghaitech/formatted_trainval_{0}/shanghaitech_part_{0}_patches_9/val'
    # 验证集Ground_Truth目录
    VAL_GT_PATH = '/opt/dataset/crowd_counting/shanghaitech/formatted_trainval_{0}/shanghaitech_part_{0}_patches_9/val_den'

    # 测试集目录
    TEST_PATH = '/opt/dataset/crowd_counting/shanghaitech/original/part_{}_final/test_data/images/'
    # 测试集Ground_Truth目录
    TEST_GT_PATH = '/opt/dataset/crowd_counting/shanghaitech/original/part_{}_final/test_data/ground_truth_csv/'

    # 测试集Ground_Truth heatmap目录
    HM_GT_PATH = './heatmaps_gt'

    EPOCHS = 200
    TRAIN_BATCH_SIZE = 32
    VAL_BATCH_SIZE = 1


current_config = LocalConfig()
