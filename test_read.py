#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software: pycharm
@file: test_read
@time: 2019/1/6 11:09
@desc:
'''

import multiprocessing
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_operation import SaveFile, LoadFile
from sklearn.preprocessing import Imputer
from Classifiers import MultiClassifiers

# normalization = lambda data: (data - np.min(data, axis= 0)) / (np.max(data, axis= 0) - np.min(data, axis= 0))
# p_1 = r'F:\GraduateDesigning\featrure\original_data_Label1_features.xlsx'
# dataset = pd.read_excel(p_1)
# dataset = np.array(dataset)
# np.random.shuffle(dataset)
# dataset = dataset[:1250, :]
# #归一化
# dataset = normalization(dataset)
# dataset_1 = np.hstack((dataset, np.ones(shape= (dataset.shape[0], 1))))
# SaveFile(data= dataset_1, savepickle_p= r'F:\GraduateDesigning\feature_t-SNE\data_1.pickle')
# print(dataset_1.shape)
# for num in range(2, 9):
#     p = r'F:\GraduateDesigning\featrure\original_data_Label%s_features.txt' % num
#     dataset = np.loadtxt(p)
#     np.random.shuffle(dataset)
#     dataset = dataset[:1250, :]
#     dataset = normalization(dataset)
#     dataset = np.hstack((dataset, np.ones(shape= (dataset.shape[0], 1))*num))
#     SaveFile(data= dataset, savepickle_p= r'F:\GraduateDesigning\feature_t-SNE\data_%s.pickle' % num)
#     print(dataset.shape)
#
# #制作降维后数据并存储（带标签存储）
# data_all = np.zeros(shape= (1, 1))
# for i in range(1, 9):
#     p = r'F:\GraduateDesigning\feature_t-SNE\data_%s.pickle' % i
#     data = LoadFile(p= p)
#     #截取特征
#     data = data[:, :191]
#     if not data_all.any():
#         data_all = data
#     else:
#         data_all = np.vstack((data_all, data))
#
# print('总数据维度为:', data_all.shape)
# #缺失值处理
# imp = Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=True)
# dataset_sim = imp.fit_transform(data_all)
# # np.random.shuffle(dataset_sim)
# #建立类
# multiclassifiers = MultiClassifiers(dataset_all=dataset_sim, dataset_sim=dataset_sim)
# tsne = MultiClassifiers.t_SNE(n_components=2, init='pca')
# #将去掉最后一列标签的数据集传入t-SNE类
# X_tsne = tsne.fit_transform(X= dataset_sim[:, :-1])
#
# # plt.figure()
# # plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
# # plt.show()
#
# #按类别保存数据
# # number = [100045, 32108, 96124, 82207, 75285, 50552, 45619]
# number = [1250 for _ in range(8)]
# row = 0
# for i in range(1, 9):
#     p= r'F:\GraduateDesigning\feature_t-SNE\tsne_data_%s.pickle' % i
#     data = np.hstack((X_tsne[row:row+number[i-1], :], np.ones(shape= (number[i-1], 1))*i))
#     row = row + number[i-1]
#     SaveFile(data= data, savepickle_p= p)
#     print('存储的降维后数据维度为:', data.shape)
#
# #绘图
# mode = ['Still', 'Walk', 'Run', 'Bike', 'Car', 'Bus', 'Train', 'Subway']
# color = ['b', 'g', 'c', 'm', 'olive', 'darkgoldenrod', 'black', 'r']
# plt.figure('20000 points-2D')
# for i in range(1, 9):
#     p = r'F:\GraduateDesigning\feature_t-SNE\tsne_data_%s.pickle' % i
#     X_tsne = LoadFile(p)
#     #去除最后一列标签
#     X_tsne = X_tsne[:, :-1] #shape= (x, y)
#     X_tsne = normalization(X_tsne)
#     plt.scatter(x= X_tsne[:, 0], y= X_tsne[:, 1], color= color.pop(), label= mode.pop())
#
# plt.xticks([])
# plt.yticks([])
# plt.legend()
# plt.show()

p = r'F:\GraduateDesigning\t_SNE_data_prune.pickle'
data = LoadFile(p)
print(data.shape)
plt.figure()
plt.scatter(data[:, 0], data[:, 1], s= 11)
plt.show()
















