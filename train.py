# -*- coding:utf-8 -*-
"""
   File Name:     train.py
   Description:   训练
   Author:        steven.yi
   date:          2019/05/05
"""
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from model import MSCNN
from src.data_loader import DataLoader
from src.metrics import mae, mse
from src.losses import l2
from config import current_config as cfg
import os
import argparse
import tensorflow as tf
import keras


def set_gpu_growth():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    cfg = tf.ConfigProto(allow_soft_placement=True)  # because no supported kernel for GPU devices is available
    cfg.gpu_options.allow_growth = True
    session = tf.Session(config=cfg)
    keras.backend.set_session(session)


def main(args):
    set_gpu_growth()
    dataset = args.dataset  # 'A' or 'B'

    train_path = cfg.TRAIN_PATH.format(dataset)
    train_gt_path = cfg.TRAIN_GT_PATH.format(dataset)
    val_path = cfg.VAL_PATH.format(dataset)
    val_gt_path = cfg.VAL_GT_PATH.format(dataset)
    # 加载数据
    print('[INFO] Loading data, wait a moment...')
    train_data_gen = DataLoader(train_path, train_gt_path, shuffle=True, gt_downsample=True)
    val_data_gen = DataLoader(val_path, val_gt_path, shuffle=False, gt_downsample=True)

    # 定义模型
    input_shape = (None, None, 1)
    model = MSCNN(input_shape)
    # 编译
    sgd = SGD(lr=1e-2, momentum=0.9, decay=0.0005)
    model.compile(optimizer=sgd, loss='mse', metrics=[mae, mse])
    model.summary()
    # 定义callback
    checkpointer_best_train = ModelCheckpoint(
        filepath=os.path.join(cfg.MODEL_DIR, 'mscnn_' + dataset + '_train.hdf5'),
        monitor='loss', verbose=1, save_best_only=True, mode='min'
    )
    lr = ReduceLROnPlateau(monitor='loss', min_lr=1e-7)
    callback_list = [checkpointer_best_train, lr]

    # 训练
    print('[INFO] Training Part_{}...'.format(dataset))
    model.fit_generator(train_data_gen.flow(cfg.TRAIN_BATCH_SIZE),
                        steps_per_epoch=train_data_gen.num_samples // cfg.TRAIN_BATCH_SIZE,
                        validation_data=val_data_gen.flow(cfg.VAL_BATCH_SIZE),
                        validation_steps=val_data_gen.num_samples // cfg.VAL_BATCH_SIZE,
                        epochs=cfg.EPOCHS,
                        callbacks=callback_list,
                        verbose=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="the dataset you want to train", choices=['A', 'B'])
    args = parser.parse_args()
    main(args)
