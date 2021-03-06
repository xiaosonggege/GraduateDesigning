#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: pointing
@time: 2019/1/8 22:46
@desc:
'''

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import manifold
from Classifiers import MultiClassifiers
from sklearn.preprocessing import Imputer
from mpl_toolkits.mplot3d import Axes3D
from data_operation import LoadFile, SaveFile
import xlrd
import pandas as pd

if __name__ == '__main__':
    mode = ['Still', 'Walk', 'Run', 'Bike', 'Car', 'Bus', 'Train', 'Subway']
    # evaluation = ['precision', 'recall', 'F-Score']
    # huawei = np.array([
    #     [0.9770, 0.1276, 0.5523],
    #     [0.9753, 0.1201, 0.5477],
    #     [0.9961, 0.1150, 0.5556],
    #     [0.9737, 0.1167, 0.5452],
    #     [0.9918, 0.1375, 0.5646],
    #     [0.9822, 0.1151, 0.5486],
    #     [0.9462, 0.1354, 0.5408],
    #     [0.9125, 0.1022, 0.5074]
    # ])
    #
    # htc = np.array([
    #     [0.9789, 0.1558, 0.5674],
    #     [0.9685, 0.1324, 0.5505],
    #     [0.9862, 0.0634, 0.5248],
    #     [0.9679, 0.0989, 0.5334],
    #     [0.9715, 0.2471, 0.6093],
    #     [0.8973, 0.0588, 0.4781],
    #     [0.9575, 0.0854, 0.5215],
    #     [0.9292, 0.1206, 0.5249]
    # ])
    #
    # acc = np.array([
    #     [0.886, 0.939, 0.799, 0.925, 0.950, 0.977, 0.975, 0.977],
    #     [0.906, 0.960, 0.945, 0.831, 0.903, 0.975, 0.995, 0.996],
    #     [0.976, 0.987, 0.986, 0.992, 0.990, 0.996, 0.999, 0.999],
    #     [0.918, 0.959, 0.884, 0.844, 0.873, 0.973, 0.999, 0.998],
    #     [0.870, 0.973, 0.696, 0.869, 0.914, 0.991, 0.991, 0.974],
    #     [0.756, 0.932, 0.655, 0.755, 0.849, 0.982, 0.999, 0.983],
    #     [0.686, 0.894, 0.585, 0.686, 0.826, 0.946, 0.799, 0.842],
    #     [0.614, 0.614, 0.464, 0.614, 0.715, 0.912, 0.806, 0.836]
    # ])
    #
    # #huawei-precision
    # eva = 0
    # picture = plt.figure(r'Sussex-Huawei Locomotion Dataset / %s' % evaluation[eva])
    # # 取一列
    # mode_eva = huawei[:, eva] * 100
    # index = np.arange(8)
    # bar_width = 0.35
    # opacity = 0.5
    # plt.bar(index, mode_eva, bar_width, alpha=opacity,
    #         color='g', label='%s' % evaluation[eva])
    # plt.xticks(index, mode)
    # plt.legend()
    # plt.grid(linestyle='--')
    # plt.ylim(0, 120)
    # for a, b in zip(index - 0.4, mode_eva):
    #     plt.text(a, b + 1.5, '%.2f%%' % b)
    #
    # #huawei-recall
    # eva = 1
    # picture = plt.figure(r'Sussex-Huawei Locomotion Dataset / %s' % evaluation[eva])
    # # 取一列
    # mode_eva = huawei[:, eva] * 100
    # index = np.arange(8)
    # bar_width = 0.35
    # opacity = 0.5
    # plt.bar(index, mode_eva, bar_width, alpha=opacity,
    #         color='r', label='%s' % evaluation[eva])
    # plt.xticks(index, mode)
    # plt.legend()
    # plt.grid(linestyle='--')
    # plt.ylim(0, 20)
    # for a, b in zip(index - 0.4, mode_eva):
    #     plt.text(a, b + 0.2, '%.2f%%' % b)
    #
    # #huawei-F-Score
    # eva = 2
    # picture = plt.figure(r'Sussex-Huawei Locomotion Dataset / %s' % evaluation[eva])
    # # 取一列
    # mode_eva = huawei[:, eva] * 100
    # index = np.arange(8)
    # bar_width = 0.35
    # opacity = 0.5
    # plt.bar(index, mode_eva, bar_width, alpha=opacity,
    #         color='b', label='%s' % evaluation[eva])
    # plt.xticks(index, mode)
    # plt.legend()
    # plt.grid(linestyle='--')
    # plt.ylim(0, 70)
    # for a, b in zip(index - 0.4, mode_eva):
    #     plt.text(a, b + 1, '%.2f%%' % b)
    #
    # #huswei-3
    # picture = plt.figure(r'Sussex-Huawei Locomotion Dataset / evaluation')
    # index = np.arange(8)
    # bar_width = 0.2
    # opacity = 0.5
    # plt.xticks(index + bar_width, mode)
    # color = ['r', 'g', 'b']
    # # 取一列
    # for eva in range(3):
    #     mode_eva = huawei[:, eva] * 100
    #     plt.bar(index + eva * bar_width, mode_eva, bar_width, alpha=opacity,
    #             color=color[eva], label='%s' % evaluation[eva])
    #     for a, b in zip(index + eva * bar_width - 0.4, mode_eva):
    #         plt.text(a, b + 1, '%.2f%%' % b)
    # plt.legend()
    # plt.grid(linestyle='--')
    # plt.ylim(0, 130)
    #
    #
    # #htc-precision
    # eva = 0
    # picture = plt.figure(r'HTC Dataset / %s' % evaluation[eva])
    # # 取一列
    # mode_eva = htc[:, eva] * 100
    # index = np.arange(8)
    # bar_width = 0.35
    # opacity = 0.8
    # plt.bar(index, mode_eva, bar_width, alpha=opacity,
    #         color='c', label='%s' % evaluation[eva])
    # plt.xticks(index, mode)
    # plt.legend()
    # plt.grid(linestyle='--')
    # plt.ylim(0, 120)
    # for a, b in zip(index - 0.4, mode_eva):
    #     plt.text(a, b + 1, '%.2f%%' % b)
    #
    # eva = 1
    # picture = plt.figure(r'HTC Dataset / %s' % evaluation[eva])
    # # 取一列
    # mode_eva = htc[:, eva] * 100
    # index = np.arange(8)
    # bar_width = 0.35
    # opacity = 0.5
    # plt.bar(index, mode_eva, bar_width, alpha=opacity,
    #         color='m', label='%s' % evaluation[eva])
    # plt.xticks(index, mode)
    # plt.legend()
    # plt.grid(linestyle='--')
    # plt.ylim(0, 30)
    # for a, b in zip(index - 0.4, mode_eva):
    #     plt.text(a, b + 0.4, '%.2f%%' % b)
    #
    # #htc-F-Score
    # eva = 2
    # picture = plt.figure(r'HTC Dataset / %s' % evaluation[eva])
    # # 取一列
    # mode_eva = htc[:, eva] * 100
    # index = np.arange(8)
    # bar_width = 0.35
    # opacity = 0.5
    # plt.bar(index, mode_eva, bar_width, alpha=opacity,
    #         color='olive', label='%s' % evaluation[eva])
    # plt.xticks(index, mode)
    # plt.legend()
    # plt.grid(linestyle='--')
    # plt.ylim(0, 70)
    # for a, b in zip(index - 0.4, mode_eva):
    #     plt.text(a, b + 1, '%.2f%%' % b)
    #
    # #htc-3
    # picture = plt.figure(r'HTC Dataset / evaluation')
    # index = np.arange(8)
    # bar_width = 0.2
    # opacity = 0.8
    # plt.xticks(index + bar_width, mode)
    # color = ['c', 'm', 'olive']
    # # 取一列
    # for eva in range(3):
    #     mode_eva = htc[:, eva] * 100
    #     plt.bar(index + eva * bar_width, mode_eva, bar_width, alpha=opacity,
    #             color=color[eva], label='%s' % evaluation[eva])
    #     for a, b in zip(index + eva * bar_width - 0.4, mode_eva):
    #         plt.text(a, b + 1, '%.2f%%' % b)
    # plt.legend()
    # plt.grid(linestyle='--')
    # plt.ylim(0, 133)
    #
    #
    # #二.
    # classifier = ['DNN', 'CNN', 'LSTM', 'Decision-Tree', 'Random-Forest', 'CNN+LSTM', 'Adaboost', 'Xgboost']
    # Accuracy = [82.6, 93.5, 74.9, 85.0, 80.0, 98.1, 96.7, 96.2]
    # picture = plt.figure(r'Accuracy-H')
    # index = np.arange(8)
    # bar_width = 0.5
    # opacity = 0.8
    # plt.xticks(index, classifier, fontsize= 5.7)
    # color = ['r', 'g', 'b', 'c', 'm', 'olive', 'darkgoldenrod', 'black']
    # plt.bar(index, Accuracy, bar_width, alpha=opacity,
    #         color=color)
    # for a, b in zip(index - 0.3, Accuracy):
    #     plt.text(a, b + 1, '%.1f%%' % b)
    # plt.ylabel('Accuracy')
    #
    # #三
    # linestyle= ['--', '-.', ':', '-', '--', '-.', ':', '-']
    # marker = ['*', 's', 'd', 'v', 'h', '8', 'p', '+']
    # index = np.arange(8)
    # picture = plt.figure('zhexian')
    # plt.xticks(index, mode)
    # color = ['r', 'g', 'b', 'c', 'm', 'olive', 'darkgoldenrod', 'black']
    # for model in range(8):
    #     plt.plot(index, acc[:, model]*100, linestyle= linestyle[model], color = color[model], label= classifier[model], marker= marker[model])
    # plt.ylabel('The Accuracy')
    # plt.legend()
    #
    # plt.show()

    #t-SNE三维绘图
    # p_1 = r'F:\GraduateDesigning\featrure\original_data_Label1_features.xlsx'
    # dataset = pd.read_excel(p_1)
    # data = np.array(dataset)
    # # np.random.shuffle(data)
    # data = data[:2500, :]
    # dataset_sim = np.hstack((data, np.ones(shape= (data.shape[0], 1))))
    # print(dataset_sim.shape)
    # for num in range(2, 9):
    #     p = r'F:\GraduateDesigning\featrure\original_data_Label%s_features.txt' % num
    #     dataset = np.loadtxt(p)
    #     # np.random.shuffle(dataset)
    #     dataset = dataset[:2500, :]
    #     data_fin = np.hstack((dataset, np.ones(shape= (dataset.shape[0], 1))*num))
    #     print(data_fin.shape)
    #     dataset_sim = np.vstack((dataset_sim, data_fin))
    # print(dataset_sim.shape)
    # #数据归一化
    # dataset_sim_feature = dataset_sim[:, :-1]
    # dataset_sim_feature = (dataset_sim_feature - np.min(dataset_sim_feature, axis= 0)) / (np.max(dataset_sim_feature, axis= 0) - np.min(dataset_sim_feature, axis= 0))
    # dataset_sim = np.hstack((dataset_sim_feature, dataset_sim[:, -1][:, np.newaxis]))
    # #缺失值处理
    # imp = Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=True)
    # dataset_sim = imp.fit_transform(dataset_sim)
    # dataset_all = dataset_sim
    # multiclassifiers = MultiClassifiers(dataset_all=dataset_all, dataset_sim=dataset_sim)
    # mode = {key: value for key, value in zip([str(i) for i in range(1, 9)], mode)}
    # # print(mode)
    # tsne = MultiClassifiers.t_SNE(n_components=3, init='pca')
    # print('将要执行t-SNE算法')
    # np.random.shuffle(dataset_sim)
    # X_tsne_1 = tsne.fit_transform(dataset_sim[:, :-1])
    # print('t-SNE算法已执行完毕')
    # #存储降维后矩阵
    # SaveFile(data= X_tsne_1, savepickle_p= r'F:\GraduateDesigning\t_SNE_data_prune.pickle')
    # #导入降维后的数据
    # X_tsne = LoadFile(p= r'F:\GraduateDesigning\t_SNE_data_prune.pickle')
    # print('导入的数据维度为:', X_tsne.shape)
    # # 归一化
    # X_tsne = (X_tsne - np.min(X_tsne, axis=0)) / (np.max(X_tsne, axis=0) - np.min(X_tsne, axis=0))
    # print(X_tsne.shape)
    # label = []
    # 绘制二维
    # plt.figure('t-SNE-2D')
    # for i in range(X_tsne.shape[0]):
    #     if dataset_sim[i, -1] == 1:
    #         color = 'r'
    #         mode_per = mode['1']
    #     elif dataset_sim[i, -1] == 2:
    #         color = 'g'
    #         mode_per = mode['2']
    #     elif dataset_sim[i, -1] == 3:
    #         color = 'c'
    #         mode_per = mode['3']
    #     elif dataset_sim[i, -1] == 4:
    #         color = 'm'
    #         mode_per = mode['4']
    #     elif dataset_sim[i, -1] == 5:
    #         color = 'olive'
    #         mode_per = mode['5']
    #     elif dataset_sim[i, -1] == 6:
    #         color = 'darkgoldenrod'
    #         mode_per = mode['6']
    #     elif dataset_sim[i, -1] == 7:
    #         color = 'black'
    #         mode_per = mode['7']
    #     else:
    #         color = 'b'
    #         mode_per = mode['8']
    #     #制作标签列表
    #     if mode_per not in label:
    #         label.append(mode_per)
    #         plt.scatter(x= X_tsne[i, 0], y= X_tsne[i, 1], s= 11, color= color, label= mode_per)
    #     else:
    #         plt.scatter(x=X_tsne[i, 0], y=X_tsne[i, 1], s= 11, color= color)
    # plt.xticks([])
    # plt.yticks([])
    # plt.legend()
    # 绘制三维
    # ax = plt.figure('t-SNE-3D').add_subplot(111, projection='3d')
    # for i in range(X_tsne.shape[0]):
    #     if dataset_sim[i, -1] == 1:
    #         color = 'r'
    #         mode_per = mode['1']
    #     elif dataset_sim[i, -1] == 2:
    #         color = 'g'
    #         mode_per = mode['2']
    #     elif dataset_sim[i, -1] == 3:
    #         color = 'c'
    #         mode_per = mode['3']
    #     elif dataset_sim[i, -1] == 4:
    #         color = 'm'
    #         mode_per = mode['4']
    #     elif dataset_sim[i, -1] == 5:
    #         color = 'olive'
    #         mode_per = mode['5']
    #     elif dataset_sim[i, -1] == 6:
    #         color = 'darkgoldenrod'
    #         mode_per = mode['6']
    #     elif dataset_sim[i, -1] == 7:
    #         color = 'black'
    #         mode_per = mode['7']
    #     else:
    #         color = 'b'
    #         mode_per = mode['8']
    #     #制作标签列表
    #     if mode_per not in label:
    #         label.append(mode_per)
    #         ax.scatter(xs=X_tsne[i, 0], ys=X_tsne[i, 1], zs=X_tsne[i, -1], s= 11, color=color, label=mode_per)
    #     else:
    #         ax.scatter(xs=X_tsne[i, 0], ys=X_tsne[i, 1], zs=X_tsne[i, -1], s= 11, color=color)
    #
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # ax.legend()
    # plt.show()

    #matrix绘图
    matrix = np.array([[0.9596, 0.9540, 0.9909, 0.9728, 0.9717, 0.9481, 0.8404, 0.8126, 0.9309],
                       [0.9165, 0.9704, 0.9893, 0.9607, 0.9662, 0.9141, 0.7853, 0.7862, 0.9101],
                       [0.9505, 0.9775, 0.9906, 0.9678, 0.9858, 0.9522, 0.8757, 0.8170, 0.9387],
                       [0.9326, 0.9615, 0.9838, 0.9691, 0.9742, 0.9559, 0.8943, 0.7874, 0.9332],
                       [0.9112, 0.9625, 0.9926, 0.9643, 0.9857, 0.9250, 0.8309, 0.8147, 0.9229],
                       [0.9770, 0.9750, 0.9960, 0.9730, 0.9910, 0.9820, 0.9460, 0.9120, 0.9810]])

    # indexs = ['a', 'b', 'c', 'd', 'e', 'f']
    # mode = ['Still', 'Walk', 'Run', 'Bike', 'Car', 'Bus', 'Train', 'Subway', 'Accuracy']
    # linestyle= ['--', '-.', ':', '-', '--', '-.', ':', '-']
    # marker = ['*', 's', 'd', 'v', 'h', '8', 'p', '+']
    color = ['r', 'g', 'b', 'c', 'm', 'olive', 'darkgoldenrod', 'black']
    # plt.figure('matrix')
    # index = range(9)
    # plt.xticks(index, mode, fontsize= 5.7)
    # for i in range(6):
    #     plt.plot(index, matrix[i, :], linestyle= linestyle[i], color= color[i], label= indexs[i], marker= marker[i])
    # plt.ylabel('The Accuracy')
    # plt.legend()
    # plt.show()

    picture = plt.figure('f')
    # 取一列
    mode_eva = matrix[-1, :-1] * 100
    index = np.arange(8)
    bar_width = 0.6
    opacity = 0.5
    plt.bar(index, mode_eva, bar_width, alpha=opacity,
            color=color)
    plt.xticks(index, mode)
    # plt.legend()
    # plt.grid(linestyle='--')
    plt.ylim(0, 110)
    for a, b in zip(index - 0.4, mode_eva):
        plt.text(a, b + 1, '%.2f%%' % b)
    plt.ylabel('Accuracy')
    plt.show()




