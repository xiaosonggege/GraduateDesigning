#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: data_point
@time: 2019/1/17 16:35
@desc:
'''
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

Adaboost = pd.read_excel(r'F:\GraduateDesigning\dataframe\Adaboost.xlsx')
XGBoost = pd.read_excel(r'F:\GraduateDesigning\dataframe\XGBoost.xlsx')
Adaboost = np.array(Adaboost)
XGBoost = np.array(XGBoost)
# SVM = pd.read_excel(r'F:\GraduateDesigning\dataframe\SVM.xlsx')
# print(Adaboost)
# print(XGBoost)

mode = ['Subway', 'Train', 'Bus', 'Car']
#Adaboost-precision
picture = plt.figure(r'Adaboost-precision')
y = Adaboost[:, 0]
index = np.arange(4)
bar_width = 0.35
opacity = 0.5
plt.bar(index, y, bar_width, alpha=opacity,
        color='olive', label='precision')
plt.xticks(index, mode)
plt.legend()
plt.grid(linestyle='--')
plt.ylim(0, 1.2)
for a, b in zip(index - 0.15, y):
    plt.text(a, b + 0.01, '%.2f%%' % b)
# plt.show()

#Adaboost-recall
picture = plt.figure(r'Adaboost-recall')
y = Adaboost[:, 1]
index = np.arange(4)
bar_width = 0.35
opacity = 0.5
plt.bar(index, y, bar_width, alpha=opacity,
        color='c', label='recall')
plt.xticks(index, mode)
plt.legend()
plt.grid(linestyle='--')
plt.ylim(0, 1.2)
for a, b in zip(index - 0.15, y):
    plt.text(a, b + 0.01, '%.2f%%' % b)
# plt.show()

#Adaboost-F1
picture = plt.figure(r'Adaboost-F1')
y = Adaboost[:, -1]
index = np.arange(4)
bar_width = 0.35
opacity = 0.5
plt.bar(index, y, bar_width, alpha=opacity,
        color='r', label='F1')
plt.xticks(index, mode)
plt.legend()
plt.grid(linestyle='--')
plt.ylim(0, 1.2)
for a, b in zip(index - 0.15, y):
    plt.text(a, b + 0.01, '%.2f%%' % b)
# plt.show()

#XGBoost-precision
picture = plt.figure(r'XGBoost-precision')
y = XGBoost[:, 0]
index = np.arange(4)
bar_width = 0.35
opacity = 0.5
plt.bar(index, y, bar_width, alpha=opacity,
        color='c', label='precision')
plt.xticks(index, mode)
plt.legend()
plt.grid(linestyle='--')
plt.ylim(0, 1.2)
for a, b in zip(index - 0.15, y):
    plt.text(a, b + 0.01, '%.2f%%' % b)
# plt.show()

#XGBoost-recall
picture = plt.figure(r'XGBoost-recall')
y = XGBoost[:, 1]
index = np.arange(4)
bar_width = 0.35
opacity = 0.5
plt.bar(index, y, bar_width, alpha=opacity,
        color='m', label='recall')
plt.xticks(index, mode)
plt.legend()
plt.grid(linestyle='--')
plt.ylim(0, 1.2)
for a, b in zip(index - 0.15, y):
    plt.text(a, b + 0.01, '%.2f%%' % b)
# plt.show()

#XGBoost-F1
picture = plt.figure(r'XGBoost-F1')
y = XGBoost[:, -1]
index = np.arange(4)
bar_width = 0.35
opacity = 0.5
plt.bar(index, y, bar_width, alpha=opacity,
        color='g', label='F1')
plt.xticks(index, mode)
plt.legend()
plt.grid(linestyle='--')
plt.ylim(0, 1.2)
for a, b in zip(index - 0.15, y):
    plt.text(a, b + 0.01, '%.2f%%' % b)
plt.show()

# #SVM-precision
# picture = plt.figure(r'SVM-precision')
# y = SVM[:, 0]
# index = np.arange(4)
# bar_width = 0.35
# opacity = 0.5
# plt.bar(index, y, bar_width, alpha=opacity,
#         color='c', label='precision')
# plt.xticks(index, mode)
# plt.legend()
# plt.grid(linestyle='--')
# plt.ylim(0, 1.2)
# for a, b in zip(index - 0.15, y):
#     plt.text(a, b + 0.01, '%.2f%%' % b)
# # plt.show()
#
# #SVM-recall
# picture = plt.figure(r'SVM-recall')
# y = SVM[:, 1]
# index = np.arange(4)
# bar_width = 0.35
# opacity = 0.5
# plt.bar(index, y, bar_width, alpha=opacity,
#         color='m', label='recall')
# plt.xticks(index, mode)
# plt.legend()
# plt.grid(linestyle='--')
# plt.ylim(0, 1.2)
# for a, b in zip(index - 0.15, y):
#     plt.text(a, b + 0.01, '%.2f%%' % b)
# # plt.show()
#
# #SVM-F1
# picture = plt.figure(r'SVM-F1')
# y = XGBoost[:, -1]
# index = np.arange(4)
# bar_width = 0.35
# opacity = 0.5
# plt.bar(index, y, bar_width, alpha=opacity,
#         color='g', label='F1')
# plt.xticks(index, mode)
# plt.legend()
# plt.grid(linestyle='--')
# plt.ylim(0, 1.2)
# for a, b in zip(index - 0.15, y):
#     plt.text(a, b + 0.01, '%.2f%%' % b)
# plt.show()

