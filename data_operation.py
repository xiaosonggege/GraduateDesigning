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
from scipy.integrate import quad
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

def EquilibriumDenoising(p_former):
    '''
    对数据进行均衡截取、去噪
    :param p_former: 数据目录前缀
    :return: 经过均衡化、去噪后的数据集
    '''

    # data为4类数据经过数据均衡、去噪后的矩阵
    data = np.zeros(shape=[125050 * 4, 11])  # 如果制作5000个shape= [62550*4, 11], 10000个shape= [125050*4, 11]
    for num in range(3, 7):
        # p = r'F:\GraduateDesigning\ICT DataSet\Label_%s.txt' % num
        p = p_former + r'\\' + r'ICT DataSet\Label_%s.txt' % num
        with open(p, 'r') as file:
            print('正在处理第%d个模型' % num)
            sub_data = np.loadtxt(file, delimiter=',', skiprows=0)[:200000, :]  # 如果制作5000时取100000, 10000时取200000
            i = 0
            while i < sub_data.shape[0]:
                if sub_data[i, :].any() == 0 or sub_data[i, -2] < 0:
                    sub_data = np.delete(sub_data, i, axis=0)
                else:
                    i += 1
                    print(i)
            sub_data = np.delete(sub_data, [3, 4, 5], axis=1)
            data = sub_data[:125051, :] if data.any() == 0 else np.vstack(
                (data, sub_data[:125051, :]))  # 如果制作5000时取62551, 10000时取125051

    return data

def LoadFile(p):
    '''读取文件'''
    data = np.array([0])
    try:
        with open(p, 'rb') as file:
            data = pickle.load(file)
    except:
        print('文件不存在!')
    finally:
        return data

class StatisticStack:
    '''
    计算一系列加速度帧统计、时域、频域特征，并组合为特征向量的转置
    '''
    __slots__ = ('__time_series', '__mean', '__var', '__std', '__median', '__min', '__max',
                 '__max_min', '__interquartile_range', '__kurtosis', '__skewness', '__rms',
                 '__integral', '__auto_corr', '__mean_cross_rate', '__DC', '__spectral_energy')

    @staticmethod
    def fft(time_series):
        '''
        对时域信号进行快速傅里叶变换
        :param time_series: 待处理的时间序列
        :return: 时间序列映射到频域后的频谱信号
        '''
        return sci.fft(time_series)

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
        self.__integral = quad(lambda x: x, self.__time_series[0], self.__time_series[-1])
        self.__auto_corr = stats.pearsonr(self.__time_series, self.__time_series)[0]
        self.__mean_cross_rate = np.sum(np.where(self.__time_series > self.__mean, 1, 0)) / \
            np.sum(self.__time_series)
        self.__spectral_energy = quad(StatisticStack.fft(time_series), StatisticStack.fft(self.__time_series)[0],
                                      StatisticStack.fft(self.__time_series)[-1])

    def feature_stack(self):
        '''
        对每个序列的所有特征组合为特征向量
        :return: 数据集中单个列按采样频率计算滑动窗口计算后的特征向量
        '''
        return np.array([self.__mean, self.__var, self.__std, self.__median, self.__min, self.__max, self.__max_min,
                         self.__interquartile_range, self.__kurtosis, self.__skewness, self.__integral, self.__auto_corr,
                         self.__mean_cross_rate, self.__spectral_energy])

def Acc_h(a, v):
    '''
    计算出去除重力加速度和在重力加速度方向上分量后的水平加速度
    :param a: 加速度计测量的三维加速度坐标
    :param v: 估算出来的三维重力加速度
    :return: 水平加速度
    '''

    #去除重力加速度后的加速度
    d = a - v
    #d在重力加速度方向上的分量
    p = ((d * v) / (v * v)) * v
    #d在水平方向上的分量
    h = d - p
    return h

def GravityEstimate(acc_data, Gest, w_g, Thvar_ori, var_Thr, inc):
    '''
    估计重力加速度在三个坐标轴上的数值
    :param acc_data: 重力加速度矩阵, shape= (len_window*4, 3) #考虑到任何分量都大于w_g
    :param Gest: 重力加速度初始值, shape= (gx, gy, gz)
    :param w_g: 窗内加速度均值和重力加速度之差的阈值
    :param Thvar_ori: 窗内加速度方差阈值
    :param var_Thr: 方差条件阈值
    :param inc: inc数值
    :return: 三个方向的重力加速度值, shape= (vx, vy, vz)
    '''
    #计算窗内各个轴上加速度均值和方差
    W_mean = np.mean(acc_data[:int(acc_data.shape[0]/4), :], axis= 1)
    W_var = np.var(acc_data[:int(acc_data.shape[0]/4), :], axis= 1)
    #如果滑动窗口内的加速度均值和估计的重力加速度相差较大，则复位方差阈值
    if (W_mean - Gest).any() >= w_g:
        THvar = Thvar_ori #bug 改！
    if W_var.any() < 1.5:
        if W_var.any() < THvar:
            Gest = W_mean
            THvar = (W_var + THvar) / 2
            VarIncrease = THvar * inc
        else:
            THvar = THvar + VarIncrease
    else:
        Gest = np.mean(acc_data, axis= 1)
    return Gest



def matrix_operation(data):
    '''
    对进行处理后的数据集进行滑动窗口特征计算，并生成数据矩阵
    :param data: 待处理数据,shape= (5000/10000, 9+1)
    :return: 数据矩阵
    '''
    dataset = np.zeros(shape= (1, 9*14)) #此处9*14需要修改为9*每列具有的所有特征数（统计+时域+频域）总和
    feature_dataset = np.zeros(shape= (1, 14)) #此处14需要修改为每列具有的所有特征数（统计+时域+频域）总和
    for i in range(0, data.shape[0], 50):
        #因为data最后一列为标签
        for j in range(data.shape[-1]-1):
            statisticstack = StatisticStack(data[i:i+100, j])
            feature_stack = statisticstack.feature_stack()
            feature_dataset = feature_stack if feature_dataset.any() == 0 else \
                np.hstack((feature_dataset, feature_stack[np.newaxis, :]))

        dataset = feature_dataset if dataset.any() == 0 else np.vstack((dataset, feature_dataset))
        feature_dataset = np.zeros(shape= (1, 14))

    #将特征矩阵和标签向量进行组合并返回
    return np.hstack((dataset, data[:, -1][np.newaxis, :]))

if __name__ == '__main__':
    data = EquilibriumDenoising(p_former=r'F:\GraduateDesigning')
    print(data)
    # SaveFile(data, savepickle_p= r'F:\GraduateDesigning\data_10000.pickle') #5000/10000





