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
from sklearn import manifold
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import model_selection

#测试代码需要
import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

class MultiClassifiers:
    __slots__ = ('__dataset_all', '__dataset_sim')

    @staticmethod
    def shuffle(data):
        '''
        打乱数据
        :param data: 待处理数据
        :return: 随机打乱后的数据
        '''
        data_shuffle = data
        np.random.shuffle(data_shuffle)
        return data_shuffle

    @classmethod
    def multi_SVM(cls, kernel, C, decision_function_shape, tol):
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

    @classmethod
    def multi_Adaboost(cls, max_depth, min_samples_split, min_samples_leaf, algorithm, n_estimators, learning_rate):
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

    @classmethod
    def multi_XGBoost(cls, max_depth, learning_rate, n_estimators, objective, nthread, gamma, min_child_weight,
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

    @classmethod
    def t_SNE(cls, n_components, perplexity=30.0, early_exaggeration=12, learning_rate=200, n_iter=1000,
              min_grad_norm=1e-7, init='pca', verbose=0, method='barnes_hut', angle=0.5):
        '''
        t-SNE降维可视化
        :param n_components: 嵌入空间的维度
        :param perpexity: 混乱度，表示t-SNE优化过程中考虑邻近点的多少，默认为30，建议取值在5到50之间
        :param early_exaggeration: 表示嵌入空间簇间距的大小，默认为12，该值越大，可视化后的簇间距越大
        :param learning_rate: 学习率，表示梯度下降的快慢，默认为200，建议取值在10到1000之间
        :param n_iter: 迭代次数，默认为1000，自定义设置时应保证大于250
        :param min_grad_norm: 如果梯度小于该值，则停止优化。默认为1e-7
        :param init: 初始化，默认为random。取值为random为随机初始化，取值为pca为利用PCA进行初始化（常用），
        取值为numpy数组时必须shape=(n_samples, n_components)
        :param verbose: 是否打印优化信息，取值0或1，默认为0=>不打印信息。打印的信息为：近邻点数量、耗时、σ、KL散度、误差等
        :param method: 两种优化方法：barnets_hut和exact。第一种耗时O(NlogN)，第二种耗时O(N^2)但是误差小，
        同时第二种方法不能用于百万级样本
        :param angle: 当method=barnets_hut时，该参数有用，用于均衡效率与误差，默认值为0.5，该值越大，效率越高&误差越大，
        否则反之。当该值在0.2-0.8之间时，无变化。
        :return: t-SNE类
        '''
        tsne = manifold.TSNE(
            n_components=n_components,
            perplexity=perplexity,
            early_exaggeration=early_exaggeration,
            learning_rate=learning_rate,
            n_iter=n_iter,
            min_grad_norm=min_grad_norm,
            init=init,
            verbose=verbose,
            random_state=32,
            method=method,
            angle=angle
        )
        return tsne

    def __init__(self, dataset_all, dataset_sim):
        '''
        分类器构造函数
        :param dataset_all: 输入6类交通模式数据
        :param dataset_sim: 输入4类交通模式数据
        '''
        self.__dataset_all = self.shuffle(dataset_all)
        self.__dataset_sim = self.shuffle(dataset_sim)

    def training_main(self, model_name, model, Threshold=None):
        '''
        针对多个模型进行训练操作
        :param model_name: 模型名称
        :param model: 需要训练的模型
        :param training_data: 需要载入的数据集
        :param Threshold: type= (T_pre, T_rec, T_F1), 精确率、召回率和F1指标阈值
        :return: None
        '''
        #初始化准确率、召回率、F1指数
        precision_rate = 0
        recall_rate = 0
        F1_rate = 0
        # k-fold对象,用于生成训练集和交叉验证集数据
        kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=32)
        # 交叉验证次数序号
        fold = 1

        for train_data_index, cv_data_index in kf.split(self.__dataset_sim):
            # 找到对应索引数据
            train_data, cv_data = self.__dataset_all[train_data_index], self.__dataset_all[cv_data_index]

            # 训练数据
            model.fit(X=train_data[:, :-1], y=train_data[:, -1])

            print('第%s折模型训练集精度为: %s' % (fold, model.score(train_data[:, :-1], train_data[:, -1])))

            # 对验证集进行预测
            pred_cv = model.predict(cv_data[:, :-1])
            # 对验证数据进行指标评估
            # precision_rate = ((fold - 1) * precision_rate + precision_score(cv_data[:-1], pred_cv)) / fold
            # recall_rate = ((fold - 1) * recall_rate + recall_score(cv_data[:-1], pred_cv)) / fold
            # F1_rate = ((fold - 1) * F1_rate + f1_score(cv_data[:-1], pred_cv)) / fold

            # print(cv_data[:, -1].shape, pred_cv.shape)
            # print(precision_score(cv_data[:-1], pred_cv))
            print(model.score(cv_data[:, :-1], cv_data[:, -1]))

            fold += 1

        # print('模型 %s在验证集上的性能指标为: 准确率- %.8f, 召回率- %.8f, F1指标- %.8f' %
        #         (model_name, precision_rate, recall_rate, F1_rate))


if __name__ == '__main__':
    # dataset = np.arange(50)
    # multiclassifier = MultiClassifiers(dataset= dataset)
    # digits = load_digits(n_class= 10)
    # x, y = digits.data, digits.target
    # #初始化image, image_h矩阵
    # image = np.zeros(shape= (8, 8))
    # image_h = np.zeros(shape= (8, 8*20))
    # num = 0
    # for i in range(20):
    #     for j in range(20):
    #         image = digits.images[num] if image.any() == 0 else np.hstack((image, digits.images[num]))
    #         num += 1
    #     # print(image.shape, image_h.shape)
    #     image_h = image if image_h.any() == 0 else np.vstack((image_h, image))
    #     image = np.zeros(shape= (8, 8))

    # print(x.shape, y.shape)
    # print(type(digits.images[0]))
    # figure = plt.figure()
    # plt.imshow(image_h, cmap=plt.cm.binary)
    # figure_1 = plt.figure()
    # plt.imshow(digits.images[2], cmap=plt.cm.binary)
    # plt.show()

    # tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    # X_tsne = tsne.fit_transform(x)
    # print(X_tsne.shape)
    # x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    # X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    # plt.figure(figsize=(8, 8))
    # for i in range(X_norm.shape[0]):
    #     plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
    #              fontdict={'weight': 'bold', 'size': 9})
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()
    classifier = MultiClassifiers(dataset_all= np.arange(20), dataset_sim= np.arange(30))






