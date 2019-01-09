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
import pandas

# index = np.arange(5)
# bar_width = 0.35
# opacity = 0.4
# means_men = (20, 35, 30, 35, 27)
# means_man = (30, 20, 40, 10, 2)
# figure_1 = plt.figure()
# # plt.bar(index, means_men, bar_width, alpha=opacity,
# #         color= ['r', 'g', 'b', 'c', 'm'], label= 'Men')
# plt.plot(index, means_men, color= 'r', label= 'Men', marker= 'o')
# plt.plot(index, means_man, color= 'g', label= 'Man', marker= '>')
# plt.xlabel('group')
# plt.ylabel('scores')
# # plt.xticks(index , ('A', 'B', 'C', 'D', 'E'))
# plt.legend()
# plt.grid(linestyle= '--')
# plt.ylim(0, 60)
# for a, b in zip(index-0.05, means_men):
#     plt.text(a, b+1, '%.0f' % b)
# plt.show()

if __name__ == '__main__':
    mode = ['Still', 'Walk', 'Run', 'Bike', 'Car', 'Bus', 'Train', 'Subway']
    evaluation = ['precision', 'recall', 'F-Score']
    huawei = np.array([
        [0.9770, 0.1276, 0.5523],
        [0.9753, 0.1201, 0.5477],
        [0.9961, 0.1150, 0.5556],
        [0.9737, 0.1167, 0.5452],
        [0.9918, 0.1375, 0.5646],
        [0.9822, 0.1151, 0.5486],
        [0.9462, 0.1354, 0.5408],
        [0.9125, 0.1022, 0.5074]
    ])

    htc = np.array([
        [0.9789, 0.1558, 0.5674],
        [0.9685, 0.1324, 0.5505],
        [0.9862, 0.0634, 0.5248],
        [0.9679, 0.0989, 0.5334],
        [0.9715, 0.2471, 0.6093],
        [0.8973, 0.0588, 0.4781],
        [0.9575, 0.0854, 0.5215],
        [0.9292, 0.1206, 0.5249]
    ])

    acc = np.array([
        [0.886, 0.939, 0.799, 0.925, 0.950, 0.977, 0.975, 0.977],
        [0.906, 0.960, 0.945, 0.831, 0.903, 0.975, 0.995, 0.996],
        [0.976, 0.987, 0.986, 0.992, 0.990, 0.996, 0.999, 0.999],
        [0.918, 0.959, 0.884, 0.844, 0.873, 0.973, 0.999, 0.998],
        [0.870, 0.973, 0.696, 0.869, 0.914, 0.991, 0.991, 0.974],
        [0.756, 0.932, 0.655, 0.755, 0.849, 0.982, 0.999, 0.983],
        [0.686, 0.894, 0.585, 0.686, 0.826, 0.946, 0.799, 0.842],
        [0.614, 0.614, 0.464, 0.614, 0.715, 0.912, 0.806, 0.836]
    ])

    #huawei-precision
    eva = 0
    picture = plt.figure(r'Sussex-Huawei Locomotion Dataset / %s' % evaluation[eva])
    # 取一列
    mode_eva = huawei[:, eva] * 100
    index = np.arange(8)
    bar_width = 0.35
    opacity = 0.5
    plt.bar(index, mode_eva, bar_width, alpha=opacity,
            color='g', label='%s' % evaluation[eva])
    plt.xticks(index, mode)
    plt.legend()
    plt.grid(linestyle='--')
    plt.ylim(0, 120)
    for a, b in zip(index - 0.4, mode_eva):
        plt.text(a, b + 1.5, '%.2f%%' % b)

    #huawei-recall
    eva = 1
    picture = plt.figure(r'Sussex-Huawei Locomotion Dataset / %s' % evaluation[eva])
    # 取一列
    mode_eva = huawei[:, eva] * 100
    index = np.arange(8)
    bar_width = 0.35
    opacity = 0.5
    plt.bar(index, mode_eva, bar_width, alpha=opacity,
            color='r', label='%s' % evaluation[eva])
    plt.xticks(index, mode)
    plt.legend()
    plt.grid(linestyle='--')
    plt.ylim(0, 20)
    for a, b in zip(index - 0.4, mode_eva):
        plt.text(a, b + 0.2, '%.2f%%' % b)

    #huawei-F-Score
    eva = 2
    picture = plt.figure(r'Sussex-Huawei Locomotion Dataset / %s' % evaluation[eva])
    # 取一列
    mode_eva = huawei[:, eva] * 100
    index = np.arange(8)
    bar_width = 0.35
    opacity = 0.5
    plt.bar(index, mode_eva, bar_width, alpha=opacity,
            color='b', label='%s' % evaluation[eva])
    plt.xticks(index, mode)
    plt.legend()
    plt.grid(linestyle='--')
    plt.ylim(0, 70)
    for a, b in zip(index - 0.4, mode_eva):
        plt.text(a, b + 1, '%.2f%%' % b)

    #huswei-3
    picture = plt.figure(r'Sussex-Huawei Locomotion Dataset / evaluation')
    index = np.arange(8)
    bar_width = 0.2
    opacity = 0.5
    plt.xticks(index + bar_width, mode)
    color = ['r', 'g', 'b']
    # 取一列
    for eva in range(3):
        mode_eva = huawei[:, eva] * 100
        plt.bar(index + eva * bar_width, mode_eva, bar_width, alpha=opacity,
                color=color[eva], label='%s' % evaluation[eva])
        for a, b in zip(index + eva * bar_width - 0.4, mode_eva):
            plt.text(a, b + 1, '%.2f%%' % b)
    plt.legend()
    plt.grid(linestyle='--')
    plt.ylim(0, 130)


    #htc-precision
    eva = 0
    picture = plt.figure(r'HTC Dataset / %s' % evaluation[eva])
    # 取一列
    mode_eva = htc[:, eva] * 100
    index = np.arange(8)
    bar_width = 0.35
    opacity = 0.8
    plt.bar(index, mode_eva, bar_width, alpha=opacity,
            color='c', label='%s' % evaluation[eva])
    plt.xticks(index, mode)
    plt.legend()
    plt.grid(linestyle='--')
    plt.ylim(0, 120)
    for a, b in zip(index - 0.4, mode_eva):
        plt.text(a, b + 1, '%.2f%%' % b)

    eva = 1
    picture = plt.figure(r'HTC Dataset / %s' % evaluation[eva])
    # 取一列
    mode_eva = htc[:, eva] * 100
    index = np.arange(8)
    bar_width = 0.35
    opacity = 0.5
    plt.bar(index, mode_eva, bar_width, alpha=opacity,
            color='m', label='%s' % evaluation[eva])
    plt.xticks(index, mode)
    plt.legend()
    plt.grid(linestyle='--')
    plt.ylim(0, 30)
    for a, b in zip(index - 0.4, mode_eva):
        plt.text(a, b + 0.4, '%.2f%%' % b)

    #htc-F-Score
    eva = 2
    picture = plt.figure(r'HTC Dataset / %s' % evaluation[eva])
    # 取一列
    mode_eva = htc[:, eva] * 100
    index = np.arange(8)
    bar_width = 0.35
    opacity = 0.5
    plt.bar(index, mode_eva, bar_width, alpha=opacity,
            color='olive', label='%s' % evaluation[eva])
    plt.xticks(index, mode)
    plt.legend()
    plt.grid(linestyle='--')
    plt.ylim(0, 70)
    for a, b in zip(index - 0.4, mode_eva):
        plt.text(a, b + 1, '%.2f%%' % b)

    #htc-3
    picture = plt.figure(r'HTC Dataset / evaluation')
    index = np.arange(8)
    bar_width = 0.2
    opacity = 0.8
    plt.xticks(index + bar_width, mode)
    color = ['c', 'm', 'olive']
    # 取一列
    for eva in range(3):
        mode_eva = htc[:, eva] * 100
        plt.bar(index + eva * bar_width, mode_eva, bar_width, alpha=opacity,
                color=color[eva], label='%s' % evaluation[eva])
        for a, b in zip(index + eva * bar_width - 0.4, mode_eva):
            plt.text(a, b + 1, '%.2f%%' % b)
    plt.legend()
    plt.grid(linestyle='--')
    plt.ylim(0, 133)


    #二.
    classifier = ['DNN', 'CNN', 'LSTM', 'Decision-Tree', 'Random-Forest', 'CNN+LSTM', 'Adaboost', 'Xgboost']
    Accuracy = [82.6, 93.5, 74.9, 85.0, 80.0, 98.1, 96.7, 96.2]
    picture = plt.figure(r'Accuracy-H')
    index = np.arange(8)
    bar_width = 0.5
    opacity = 0.8
    plt.xticks(index, classifier, fontsize= 5.7)
    color = ['r', 'g', 'b', 'c', 'm', 'olive', 'darkgoldenrod', 'black']
    plt.bar(index, Accuracy, bar_width, alpha=opacity,
            color=color)
    for a, b in zip(index - 0.3, Accuracy):
        plt.text(a, b + 1, '%.1f%%' % b)
    plt.ylabel('Accuracy')

    #三
    linestyle= ['--', '-.', ':', '-', '--', '-.', ':', '-']
    marker = ['*', 's', 'd', 'v', 'h', '8', 'p', '+']
    index = np.arange(8)
    picture = plt.figure('zhexian')
    plt.xticks(index, mode)
    color = ['r', 'g', 'b', 'c', 'm', 'olive', 'darkgoldenrod', 'black']
    for model in range(8):
        plt.plot(index, acc[:, model]*100, linestyle= linestyle[model], color = color[model], label= classifier[model], marker= marker[model])
    plt.ylabel('The Accuracy')
    plt.legend()

    plt.show()
