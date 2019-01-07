#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software: pycharm
@file: test
@time: 2019/1/3 21:26
@desc:
'''

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from xgboost import plot_importance
from matplotlib import pyplot as plt
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

import numpy as np

def training_main(model_name, model, training_data, Threshold= None):
    '''
    针对多个模型进行训练操作
    :param model_name: 模型名称
    :param model: 需要训练的模型
    :param training_data: 需要载入的数据集
    :param Threshold: type= (T_pre, T_rec, T_F1), 精确率、召回率和F1指标阈值
    :return: None
    '''

    # k-fold对象,用于生成训练集和交叉验证集数据
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=32)
    #交叉验证次数序号
    fold = 1

    for train_data_index, cv_data_index in kf.split(training_data):
        # 找到对应索引数据
        train_data, cv_data = training_data[train_data_index], training_data[cv_data_index]
        # 训练数据
        model.fit(X=train_data[:, :4], y=train_data[:, -1])

        # 对验证集进行预测
        pred_cv = model.predict(cv_data[:, :4])
        #对验证数据进行指标评估
        precision_rate = 0 if fold == 1 else ((fold - 1) * precision_rate + precision_score(cv_data[:-1], pred_cv)) / fold
        recall_rate = 0 if fold == 1 else ((fold - 1) * recall_rate + recall_score(cv_data[:-1], pred_cv)) / fold
        F1_rate = 0 if fold == 1 else ((fold - 1) * F1_rate + f1_score(cv_data[:-1], pred_cv)) / fold

    print('模型 %s在验证集上的性能指标为: 准确率- %.8f, 召回率- %.8f, F1指标- %.8f' %
          (model_name, precision_rate, recall_rate, F1_rate))

if __name__ == '__main__':
    training_main()


