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
import numpy as np

def training_main(model, training_data, Evaluation_index):
    '''
    针对多个模型进行训练操作
    :param model: 需要训练的模型
    :param training_data: 需要载入的数据集
    :param Evaluation_index: 评价指标， type= tuple
    :return: None
    '''

    # k-fold对象,用于生成训练集和交叉验证集数据
    kf = model_selection.KFold(n_splits=5, shuffle=False, random_state=32)

    while 1:

        for train_data_index, cv_data_index in kf.split(training_data):
            # 找到对应索引数据
            train_data, cv_data = training_data[train_data_index], training_data[cv_data_index]
            # 训练数据
            model.fit(X=train_data[:, :4], y=train_data[:, -1])

            # 对验证集进行预测
            pred_cv = model.predict(cv_data[:, :4])



        print('CART树个数: %s, 验证集MSE: %s' % (model.n_estimators, MSE))
        X = [1] if X == [] else X + [X[-1] + 1]
        Y.append(MSE)
        if MSE < Threshold:
            break
        else:
            MSE, fold = 0, 1
            # 如果验证集MSE值大于阈值则将GBDT中弱学习器数量自增1
            model.n_estimators += 1
