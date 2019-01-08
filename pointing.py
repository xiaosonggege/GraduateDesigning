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