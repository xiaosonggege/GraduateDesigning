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
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from Classifiers import MultiClassifiers
from data_operation import LoadFile
import xgboost as xgb
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
        digraph = xgb.to_graphviz(XGBoost, num_trees=2)
        digraph.format = 'png'
        digraph.view('./traffic_xgb')
        xgb.plot_importance(XGBoost)
        plt.show()

    else:
        #t-SNE降维可视化
        tsne = MultiClassifiers.t_SNE(n_components=2, init='pca')
        X_tsne = tsne.fit_transform(dataset_sim[:, :-1])
        #归一化
        X_tsne = (X_tsne - np.min(X_tsne, axis= 0)) / (np.max(X_tsne, axis= 0) - np.min(X_tsne, axis= 0))
        print(X_tsne.shape)
        #绘制二维
        plt.figure('t-SNE-2D')
        for i in range(X_tsne.shape[0]):
            if dataset_sim[i, -1] == 3:
                color= 'r'
            elif dataset_sim[i, -1] == 4:
                color= 'g'
            elif dataset_sim[i, -1] == 5:
                color= 'c'
            else:
                color= 'm'
            plt.scatter(x= X_tsne[i, 0], y= X_tsne[i, -1], color= color)
        plt.xticks([])
        plt.yticks([])
        #绘制三维
        ax = plt.figure('t-SNE-3D').add_subplot(111, projection = '3d')
        for i in range(X_tsne.shape[0]):
            if dataset_sim[i, -1] == 3:
                color= 'r'
            elif dataset_sim[i, -1] == 4:
                color= 'g'
            elif dataset_sim[i, -1] == 5:
                color= 'c'
            else:
                color= 'm'
            ax.scatter(xs=X_tsne[i, 0], ys=X_tsne[i, 1], zs=X_tsne[i, -1], color=color)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

if __name__ == '__main__':
    #加载数据
    dataset_all = LoadFile(p= r'F:\GraduateDesigning\data\data_all.pickle')
    dataset_sim = LoadFile(p= r'F:\GraduateDesigning\data\data_sim.pickle')
    np.random.shuffle(dataset_all)
    np.random.shuffle(dataset_sim)
    # print(dataset_all[:2, -1])
    # data_all = pd.DataFrame(data= dataset_all)
    # print(np.isnan(dataset_all).any())
    # print(dataset_all.dtype)
    #对SVM分类器进行十折交叉验证
    model_main(dataset_all= dataset_all, dataset_sim= dataset_sim, operation= 'Adaboost')





