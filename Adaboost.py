#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software: pycharm
@file: Adaboost
@time: 2019/1/4 20:41
@desc:
'''

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
x, y = iris.data, iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state= 32)

clf = DecisionTreeClassifier(max_depth= 2, min_samples_split= 15, min_samples_leaf= 10)
# clf.fit(x_train, y_train)
# acc = clf.score(x_test, y_test)
# print(acc)
# predict = clf.predict(x_test)
# print(predict == y_test)

bdt = AdaBoostClassifier(clf, algorithm= 'SAMME', n_estimators= 200, learning_rate= 0.8)
bdt.fit(x_train, y_train)
acc = bdt.score(x_test, y_test)
print(acc)


