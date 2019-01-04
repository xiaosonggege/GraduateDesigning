#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: XGBoost
@time: 2019/1/4 23:09
@desc:
'''

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot as plt
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

iris = load_iris()
x, y = iris.data, iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state= 32, shuffle= True, test_size= 0.333)
model = XGBClassifier(
        max_depth=10,  # 树的最大深度(可调)
        learning_rate=0.1,  # 学习率(可调)
        n_estimators= 2,  # 树的个数
        objective='multi:softmax',  # 损失函数类型
        nthread=4,  # 线程数
        gamma=0.1,  # 节点分裂时损失函数所需最小下降值（可调）
        min_child_weight=1,  # 叶子结点最小权重
        subsample=1.,  # 随机选择样本比例建立决策树
        colsample_bytree=1.,  # 随机选择样本比例建立决策树
        reg_lambda=0,  # 二阶范数正则化项权衡值（可调）
        scale_pos_weight=1.,  # 解决样本个数不平衡问题
        random_state=1000,  # 随机种子设定值
    )

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
predict = model.predict(x_test)
print(acc)
print(predict == y_test)

digraph = xgb.to_graphviz(model, num_trees=1)
digraph.format = 'png'
digraph.view('./boston_xgb')