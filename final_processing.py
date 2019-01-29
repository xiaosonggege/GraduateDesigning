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
from sklearn.preprocessing import Imputer
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
        SVM = MultiClassifiers.multi_SVM(kernel= 'rbf', C= 1.0, decision_function_shape= 'ovo', tol= 1e-3, degree= 3, coef0= 1)
        multiclassifiers.training_main(model_name= 'SVM分类器', model= SVM)
    elif operation == 'Adaboost':
        #Adaboost分类器训练
        Adaboost = MultiClassifiers.multi_Adaboost(max_depth=2, min_samples_split=2, min_samples_leaf=1,
                                                   algorithm='SAMME.R', n_estimators=500, learning_rate=1e-2)
        multiclassifiers.training_main(model_name='Adaboost分类器', model=Adaboost)


    elif operation == 'XGBoost':
        #XGBoost分类器训练
        XGBoost = MultiClassifiers.multi_XGBoost(max_depth=2, learning_rate=1e-2, n_estimators=200,
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
        mode = dict([('3', 'Subway'), ('4', 'Train'), ('5', 'Bus'), ('6', 'Car')])
        tsne = MultiClassifiers.t_SNE(n_components=3, init='random')
        X_tsne = tsne.fit_transform(dataset_sim[:, :-1])
        #归一化
        X_tsne = (X_tsne - np.min(X_tsne, axis= 0)) / (np.max(X_tsne, axis= 0) - np.min(X_tsne, axis= 0))
        print(X_tsne.shape)
        label = []
        #绘制二维
        # plt.figure('t-SNE-2D')
        # for i in range(X_tsne.shape[0]):
        #     if dataset_sim[i, -1] == 3:
        #         color= 'r'
        #         mode_per = mode['3']
        #     elif dataset_sim[i, -1] == 4:
        #         color= 'g'
        #         mode_per = mode['4']
        #     elif dataset_sim[i, -1] == 5:
        #         color= 'c'
        #         mode_per = mode['5']
        #     else:
        #         color= 'm'
        #         mode_per = mode['6']
        #     if mode_per not in label:
        #         label.append(mode_per)
        #         plt.scatter(x= X_tsne[i, 0], y= X_tsne[i, -1], color= color, label= mode_per)
        #     else:
        #         plt.scatter(x=X_tsne[i, 0], y=X_tsne[i, -1], color=color)
        # plt.xticks([])
        # plt.yticks([])
        # plt.legend()
        #绘制三维
        ax = plt.figure('t-SNE-3D').add_subplot(111, projection = '3d')
        for i in range(X_tsne.shape[0]):
            if dataset_sim[i, -1] == 3:
                color= 'r'
                mode_per = mode['3']
            elif dataset_sim[i, -1] == 4:
                color= 'g'
                mode_per = mode['4']
            elif dataset_sim[i, -1] == 5:
                color= 'c'
                mode_per = mode['5']
            else:
                color= 'm'
                mode_per = mode['6']
            if mode_per not in label:
                label.append(mode_per)
                ax.scatter(xs=X_tsne[i, 0], ys=X_tsne[i, 1], zs=X_tsne[i, -1], color=color, label= mode_per)
            else:
                ax.scatter(xs=X_tsne[i, 0], ys=X_tsne[i, 1], zs=X_tsne[i, -1], color=color)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.legend()
        plt.show()

if __name__ == '__main__':
    #加载数据(Adaboost, XGBoost数据集)
    dataset_sim = LoadFile(p= r'F:\GraduateDesigning\finalDataset\data_sim.pickle')
    # print(dataset_sim.shape)
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=True)
    dataset_sim = imp.fit_transform(dataset_sim)
    np.random.shuffle(dataset_sim)
    #数据归一化
    dataset_sim_feature = dataset_sim[:, :-1]
    dataset_sim_feature = (dataset_sim_feature - np.min(dataset_sim_feature, axis= 0)) / (np.max(dataset_sim_feature, axis= 0) - np.min(dataset_sim_feature, axis= 0))
    dataset_sim = np.hstack((dataset_sim_feature, dataset_sim[:, -1][:, np.newaxis]))
    # print(dataset_sim.shape, dataset_sim[:, -1])
    #针对SVM制作数据集（制作待删除列索引）
    # delete_array_1 = [14, 17, 19]
    # delete_array = [14, 17, 19]
    # i = 1
    # while ((19 + i * 19) <= 171):
    #     delete_array.extend([j + i * 19 for j in delete_array_1])
    #     i += 1
    # delete_array.extend([176, 177, 178, 179, 180, 186, 187, 188, 189])
    # dataset_sim_svm = np.delete(dataset_sim, delete_array, 1)
    ###########################
    # print(dataset_all[:2, -1])
    # data_all = pd.DataFrame(data= dataset_all)
    # print(np.isnan(dataset_all).any())
    # print(dataset_all.dtype)
    #对SVM分类器进行十折交叉验证
    model_main(dataset_all= dataset_sim, dataset_sim= dataset_sim, operation= 't-SNE')
    # model_main(dataset_all= dataset_sim_svm, dataset_sim= dataset_sim_svm, operation= 'SVM')






