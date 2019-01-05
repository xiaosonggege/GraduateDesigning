#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software: pycharm
@file: data_operation
@time: 2019/1/5 17:55
@desc:
'''
import numpy as np
import scipy as sci
from scipy import stats
import pickle
import os

def SaveFile(data, savepickle_p):
    '''
    存储整理好的数据'
    :param data: 待存储数据
    :param savepickle_p: pickle后缀文件存储绝对路径
    :return: None
    '''''
    if not os.path.exists(savepickle_p):
        with open(savepickle_p, 'wb') as file:
            pickle.dump(data, file)

#data为4类数据经过数据均衡、去噪后的矩阵
data = np.ones(shape= [6250*4, 11])
for num in range(3, 7):
    p = r'D:\GraduateDesigning\ICT DataSet\Label_%s.txt' % num
    with open(p, 'r') as file:
        sub_data = np.loadtxt(file, delimiter=',', skiprows=0)
        i = 0
        while i <= sub_data.shape[0]:
            if sub_data[i, :].any() == 0 or sub_data[i, -2] < 0:
                sub_data = np.delete(sub_data, i, axis= 0)
            else:
                i += 1
        sub_data = np.delete(sub_data, [3, 4, 5], axis= 1)
        data = sub_data[:6251, :] if data.any() == 0 else np.vstack((data, sub_data[:6251, :]))

SaveFile(data, savepickle_p= r'D:\GraduateDesigning\data.pickle')

class StatisticStack:
    '''
    计算一系列加速度帧统计、时域、频域特征，并组合为特征向量的转置
    '''
    __slots__ = ('__time_series', '__mean', '__var', '__std', '__median', '__min', '__max',
                 '__max_min', '__interquartile_range', '__kurtosis', '__skewness', '__rms',
                 '__integral', '__double_integral', '__auto_corr', '__mean_cross_rate',
                 '__DC', '__spectral_energy', '__spectral_entropy', '__wavelet_entropy',
                 '__wavelet_magnitude')

    @staticmethod
    def fft(time_series):
        '''
        对时域信号进行快速傅里叶变换
        :param time_series: 待处理的时间序列
        :return: 时间序列映射到频域后的频谱信号
        '''
        return sci.fft(time_series)

    @staticmethod
    def wavelet(time_series):
        '''
        对时域信号进行小波变换
        :param time_series: 待处理的时间序列
        :return: 时间序列映射到小波变换域
        '''
        pass

    def __init__(self, time_series):
        '''
        构造函数
        :param time_series: 待处理序列
        '''
        self.__time_series = time_series
        self.__mean = np.mean(self.__time_series)
        self.__var = np.var(self.__time_series)
        self.__std = np.std(self.__time_series)
        self.__median = np.median(self.__time_series)
        self.__min = np.min(self.__time_series)
        self.__max = np.max(self.__time_series)
        self.__max_min = self.__max - self.__min
        self.__interquartile_range = np.quantile(self.__time_series, 0.75) - \
            np.quantile(self.__time_series, 0.25)
        self.__kurtosis = stats.kurtosis(self.__time_series)
        self.__skewness = stats.skew(self.__time_series)
        self.__integral = sci.


