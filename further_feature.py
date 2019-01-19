#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: further_feature
@time: 2019/1/19 21:15
@desc:
'''

import numpy as np
import matplotlib.pyplot as plt
from numpy import random as ra

rng = ra.RandomState(0)
a = rng.random_integers(low= 1, high= 3, size= (1000,))
for i in range(100, 900, 200):
    a[i:i+10] = rng.random_integers(low= 10, high= 16, size= (10,))
    if not (i % 20):
        a = -1 * a

# plt.figure('加速度图')
# plt.plot(a)
# plt.show()

#峰特征类
class Peak:
    '''峰特征提取类'''

    __slots__ = ('__scan_length', '__window_length', '__start_Th', '__end_Th', '__series')

    def __init__(self, scan_length, window_length, start_Th, end_Th, series):
        '''
        峰特征类构造函数
        :param scan_length: 扫描窗口长度
        :param window_length: 相邻滑动窗口长度
        :param start_Th: 峰起阈值（左右窗口中平均值）
        :param end_Th: 峰落阈值（左右窗口中平均值）
        :param series: 待计算的序列
        '''

        self.__scan_length = scan_length
        self.__window_length = window_length
        self.__start_Th = start_Th
        self.__end_Th = end_Th
        self.__series = series

    def find_peak_areas(self):
        '''
        找到序列中所有峰值区域
        :return: 所有峰值区域作为列表元素输出
        '''

        start = 0
        end = 0
        for i in range()


