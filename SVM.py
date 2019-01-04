#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software: pycharm
@file: SVM
@time: 2019/1/3 22:42
@desc:
'''

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import load_iris

iris = load_iris()
x, y = iris.data, iris.target

svm_traffic = SVC(
    kernel= 'rbf',  #选择核函数 ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    C= 1e3, #结构风险和经验风险之间的权衡
    decision_function_shape= 'ovo', #选择一对多（自己是一类，其他事一类，k类有k个svm）还是一对一（自己是一类，另外一个是一类，k类有k（k-1）/2个svm）
    tol= 2, #停止训练时的误差阈值
    gamma= 'auto'
)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.333, random_state= 32, shuffle= True)
svm_traffic.fit(x_train, y_train)
acc = svm_traffic.score(x_train, y_train)
print(acc)
print(type(svm_traffic))
predict = svm_traffic.predict(x_test)
print(predict == y_test)
print(y_test)




