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
from sklearn.manifold import TSNE
from Classifiers import MultiClassifiers
from data_operation import LoadFile
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

import numpy as np
import pandas as pd

def model_main(dataset_all, dataset_sim, operation):
    '''
    训练主函数
    :param dataset_all: 训练集数据全部
    :param dataset_sim: 训练集后4类数据
    :param operation: 选择训练的模型, 'SVM', 'Adaboost', 'XGBoost', 't-SNE'
    :return: None
    '''
    multiclassifiers = MultiClassifiers(dataset_all=dataset_all, dataset_sim=dataset_sim)
    if operation == 'SVM':
        #SVM分类器训练
        SVM = MultiClassifiers.multi_SVM(kernel= 'linear', C= 2, decision_function_shape= 'ovo', tol= 1e-1)
        multiclassifiers.training_main(model_name= 'SVM分类器', model= SVM)
    elif operation == 'Adaboost':
        #Adaboost分类器训练
        Adaboost = MultiClassifiers.multi_Adaboost(max_depth=2, min_samples_split=2, min_samples_leaf=1,
                                                   algorithm='SAMME.R', n_estimators=100, learning_rate=1e-2)
        multiclassifiers.training_main(model_name='Adaboost分类器', model=Adaboost)

    elif operation == 'XGBoost':
        #XGBoost分类器训练
        XGBoost = MultiClassifiers.multi_XGBoost(max_depth=2, learning_rate=1e-2, n_estimators=100,
                                                 objective='binary:logistic', nthread=4, gamma=0.1,
                                                 min_child_weight=1, subsample=1, reg_lambda=2, scale_pos_weight=1.)
        multiclassifiers.training_main(model_name='XGBoost分类器', model=XGBoost)

    else:
        #t-SNE降维可视化
        tsne = MultiClassifiers.t_SNE(n_components=2, init='pca')
        X_tsne = tsne.fit_transform(dataset_sim[:, :-1])
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
        plt.figure('t-SNE')
        for i in range(X_norm.shape[0]):
            plt.text(X_norm[i, 0], X_norm[i, 1], str(dataset_sim[:, -1][i]), color=plt.cm.Set1(dataset_sim[:, -1][i]),
                     fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.show()



def demention(t_sne, dataset):
    '''
    对数据进行可视化
    :param t_sne: t-SNE降维类型
    :param dataset: 待可视化数据集
    :return: None
    '''
    pass

def cdf_picture(statistic_data, data_name):
    '''
    对统计量进行CDF绘图
    :param statistic_data: 待统计数据
    :param data_name: 统计量名称
    :return: None
    '''
    picture = plt.figure(data_name)
    hist = plt.hist(x= statistic_data, bins= 100, normed= True, cumulative= True, histtype= 'step')
    plt.show()


if __name__ == '__main__':
    #加载数据
    dataset_all = LoadFile(p= r'F:\GraduateDesigning\data_all.pickle')
    dataset_sim = LoadFile(p= r'F:\GraduateDesigning\data_sim.pickle')
    np.random.shuffle(dataset_all)
    np.random.shuffle(dataset_sim)
    # print(dataset_all[:2, -1])
    # data_all = pd.DataFrame(data= dataset_all)
    # print(np.isnan(dataset_all).any())
    # print(dataset_all.dtype)
    #对SVM分类器进行十折交叉验证
    model_main(dataset_all= dataset_all, dataset_sim= dataset_sim, operation= 'Adaboost')





