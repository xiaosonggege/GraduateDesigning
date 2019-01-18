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
from data_operation import LoadFile
from scipy import stats

Adaboost = pd.read_excel(r'F:\GraduateDesigning\dataframe\Adaboost.xlsx')
XGBoost = pd.read_excel(r'F:\GraduateDesigning\dataframe\XGBoost.xlsx')
# SVM = pd.read_excel(r'F:\GraduateDesigning\dataframe\SVM.xlsx')
Adaboost = np.array(Adaboost)
XGBoost = np.array(XGBoost)
# print(Adaboost)
# print(XGBoost)

mode = ['Subway', 'Train', 'Bus', 'Car']
#Adaboost-precision
# picture = plt.figure(r'Adaboost-precision')
# y = Adaboost[:, 0]
# index = np.arange(4)
# bar_width = 0.35
# opacity = 0.5
# plt.bar(index, y, bar_width, alpha=opacity,
#         color='olive', label='precision')
# plt.xticks(index, mode)
# plt.legend()
# plt.grid(linestyle='--')
# plt.ylim(0, 1.2)
# for a, b in zip(index - 0.15, y):
#     plt.text(a, b + 0.01, '%.2f%%' % b)
# plt.show()

#Adaboost-recall
# picture = plt.figure(r'Adaboost-recall')
# y = Adaboost[:, 1]
# index = np.arange(4)
# bar_width = 0.35
# opacity = 0.5
# plt.bar(index, y, bar_width, alpha=opacity,
#         color='c', label='recall')
# plt.xticks(index, mode)
# plt.legend()
# plt.grid(linestyle='--')
# plt.ylim(0, 1.2)
# for a, b in zip(index - 0.15, y):
#     plt.text(a, b + 0.01, '%.2f%%' % b)
# plt.show()

#Adaboost-F1
# picture = plt.figure(r'Adaboost-F1')
# y = Adaboost[:, -1]
# index = np.arange(4)
# bar_width = 0.35
# opacity = 0.5
# plt.bar(index, y, bar_width, alpha=opacity,
#         color='r', label='F1')
# plt.xticks(index, mode)
# plt.legend()
# plt.grid(linestyle='--')
# plt.ylim(0, 1.2)
# for a, b in zip(index - 0.15, y):
#     plt.text(a, b + 0.01, '%.2f%%' % b)
# plt.show()

#XGBoost-precision
# picture = plt.figure(r'XGBoost-precision')
# y = XGBoost[:, 0]
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
# plt.show()

#XGBoost-recall
# picture = plt.figure(r'XGBoost-recall')
# y = XGBoost[:, 1]
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
# plt.show()

#XGBoost-F1
# picture = plt.figure(r'XGBoost-F1')
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

# 精确度折线图
# linestyle= ['--', '-.']
# marker = ['*', 's']
# index = np.arange(4)
# picture = plt.figure('Accuracy')
# plt.xticks(index, mode)
# color = ['r', 'g']
# plt.plot(index, Adaboost[:, -1], linestyle= linestyle[0], color= color[0], label= 'Adaboost', marker= marker[0])
# plt.plot(index, XGBoost[:, -1], linestyle= linestyle[-1], color= color[-1], label= 'XGBoost', marker= marker[-1])
# plt.ylabel('The Accuracy')
# plt.legend()
#
# plt.show()

#CDF统计方法可视化

def cdf_picture(statistic_data):
    '''
    计算CDF柱状图对象
    :param statistic_data: 待统计数据
    :return: object of CumfreqResult
    '''
    hist, bin_edges = np.histogram(statistic_data)
    cdf = np.cumsum(hist)
    return cdf


# path = r'F:\GraduateDesigning\data\data_sim.pickle'
# data_sim = LoadFile(p= path)
# print(data_sim)
# plt.figure('CDF')
# linestyle = ['--', '-.', ':', '-']
# marker = ['*', 's', 'd', 'v']
# color = ['r', 'g', 'b', 'c']
# j = 0
# for i in range(0, data_sim.shape[0], 1250):
#     sub_series = data_sim[i:i+1250, -30]
#     cdf = cdf_picture(sub_series)
#     cdf = (cdf - np.min(cdf)) / (np.max(cdf) - np.min(cdf))
#     plt.plot(cdf, linestyle= linestyle[j], color= color[j], marker= marker[j], label= mode[j])
#     j += 1
# plt.ylabel('CDF')
# plt.legend()
# plt.show()
# plt.figure()
# cdf = cdf_picture(data_sim[:1250, 3])
# cdf = (cdf - np.min(cdf)) / (np.max(cdf) - np.min(cdf))
# plt.plot(cdf)
# plt.show()
# print(cdf)

