#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: Classifiers
@time: 2019/1/7 21:46
@desc:
'''

from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost import XGBClassifier

#测试代码需要
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class MultiClassifiers:
    __slots__ = ('__dataset')

    def __init__(self, dataset):
        '''
        多分类器构造函数
        :param dataset: 待处理数据集（可以为交叉验证的一部分）
        '''

        self.__dataset = dataset

    def multi_SVM(self, kernel, C, decision_function_shape, tol):
        '''
        多分类SVM分类器
        :param kernel: 选择的核函数 ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
        :param C: # 结构风险和经验风险之间的权衡
        :param decision_function_shape:  'ovo'/'ovr'
        # 选择一对多（自己是一类，其他事一类，k类有k个svm）还是一对一（自己是一类，另外一个是一类，k类有k（k-1）/2个svm）
        :param tol: # 停止训练时的误差阈值
        :return: SVM对象
        '''
        multi_svm = SVC(
            kernel= kernel,
            C= C,
            decision_function_shape= decision_function_shape,
            tol= tol,
            gamma='auto'
        )

        return multi_svm

    def multi_Adaboost(self, max_depth, min_samples_split, min_samples_leaf, algorithm, n_estimators, learning_rate):
        '''
        多分类CART树
        :param max_depth: 树最大深度
        :param min_samples_split: 继续划分叶子结点所需要的最小例子数
        :param min_samples_leaf: 叶子结点中最少要有的实例数量
        :param algorithm:  If 'SAMME.R' then use the SAMME.R real boosting algorithm.
        ``base_estimator`` must support calculation of class probabilities. If 'SAMME'
         then use the SAMME discrete boosting algorithm. The SAMME.R algorithm typically
         converges faster than SAMME, achieving a lower test error with fewer boosting iterations.
        :param n_estimators: 集成子学习器的最大数量
        :param learning_rate: 控制每类数据的分布的收缩
        :return: Adaboost对象
        '''
        #构建CART决策树作为子学习器
        clf = DecisionTreeClassifier(
            max_depth= max_depth,
            min_samples_split= min_samples_split,
            min_samples_leaf= min_samples_leaf
        )

        #构建Adaboost对象
        bdt = AdaBoostClassifier(
            base_estimator= clf,
            algorithm= algorithm,
            n_estimators= n_estimators,
            learning_rate= learning_rate
        )

        return bdt

    def multi_XGBoost(self, max_depth, learning_rate, n_estimators, objective, nthread, gamma, min_child_weight,
                      subsample, reg_lambda, scale_pos_weight):
        '''
        XGBoost对象
        :param max_depth: 树的最大深度
        :param learning_rate: 学习率
        :param n_estimators: 树的个数
        :param objective: 损失函数类型
       'reg:logistic' –逻辑回归。
       'binary:logistic' –二分类的逻辑回归问题，输出为概率。
       'binary:logitraw' –二分类的逻辑回归问题，输出的结果为wTx。
       'count:poisson' –计数问题的poisson回归，输出结果为poisson分布。在poisson回归中，max_delta_step的缺省值为0.7。(used to safeguard optimization)
       'multi:softmax' –让XGBoost采用softmax目标函数处理多分类问题，同时需要设置参数num_class（类别个数）
       'multi:softprob' –和softmax一样，但是输出的是ndata * nclass的向量，可以将该向量reshape成ndata行nclass列的矩阵。没行数据表示样本所属于每个类别的概率。
       'rank:pairwise' –set XGBoost to do ranking task by minimizing the pairwise loss
        :param nthread: 线程数
        :param gamma: 节点分裂时损失函数所需最小下降值
        :param min_child_weight: 叶子结点最小权重
        :param subsample: 随机选择样本比例建立决策树
        :param reg_lambda: 二阶范数正则化项权衡值
        :param scale_pos_weight: 解决样本个数不平衡问题
        :return: XGBoost对象
        '''
        xgbc = XGBClassifier(
            max_depth= max_depth,
            learning_rate= learning_rate,
            n_estimators= n_estimators,
            objective= objective,
            nthread= nthread,
            gamma= gamma,
            min_child_weight= min_child_weight,
            subsample= subsample,
            colsample_bytree= subsample,
            reg_lambda= reg_lambda,
            scale_pos_weight= scale_pos_weight,
            random_state= 32,
        )

        return xgbc

if __name__ == '__main__':
    dataset = np.arange(50)
    multiclassifier = MultiClassifiers(dataset= dataset)



