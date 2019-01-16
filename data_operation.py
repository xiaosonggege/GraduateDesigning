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
import pandas as pd
import pickle
import os
import multiprocessing

def SaveFile(data, savepickle_p):
    '''
    存储整理好的数据
    :param data: 待存储数据
    :param savepickle_p: pickle后缀文件存储绝对路径
    :return: None
    '''
    if not os.path.exists(savepickle_p):
        with open(savepickle_p, 'wb') as file:
            pickle.dump(data, file)

def LoadFile(p):
    '''
    读取文件
    :param p: 数据集绝对路径
    :return: 数据集
    '''
    data = np.array([0])
    try:
        with open(p, 'rb') as file:
            data = pickle.load(file)
    except:
        print('文件不存在!')
    finally:
        return data

def EquilibriumDenoising(p_former, class_num):
    '''
    对数据进行均衡截取、去噪
    :param p_former: 数据目录前缀
    :param class_num: 需要处理的模式类别
    :return: 经过均衡化、去噪后的数据集
    '''
    p = p_former + r'\\' + r'ICT DataSet\Label_%s.txt' % class_num
    with open(p, 'r') as file:
        print('正在处理第%s个模型数据' % class_num)
        #截取数据,数据标签不对所以去掉最后一列
        tru_data = np.loadtxt(file, delimiter=',', skiprows=0)[:100000, :]  # 如果制作5000时取100000, 10000时取200000
        data_frame = pd.DataFrame(tru_data[:, :-1], columns= ['acc_x', 'acc_y', 'acc_z', 'lin_acc_x', 'lin_acc_y', 'lin_acc_z',
                                            'gyr_x', 'gyr_y', 'gyr_z', 'mag_x', 'mag_y', 'mag_z', 'pre'])

        data_frame = data_frame.drop(columns=['lin_acc_x', 'lin_acc_y', 'lin_acc_z'])

        data_frame = data_frame[(~data_frame['acc_x'].isin([0])) & (~data_frame['acc_y'].isin([0])) & ((~data_frame['acc_z'].isin([0])))\
            & (~data_frame['gyr_x'].isin([0])) & (~data_frame['gyr_y'].isin([0])) & (~data_frame['gyr_z'].isin([0])) &
                                (~data_frame['mag_x'].isin([0])) & (~data_frame['mag_y'].isin([0])) & (~data_frame['mag_z'].isin([0]))]

        data_frame = data_frame.loc[data_frame['pre'] >= 0]
        #DataFrame类转ndarray类矩阵
        data_frame = data_frame.apply(lambda x: x.astype(np.float64))
        data = np.array(data_frame)[:62600, :] #62600个数据,多取50个
        data = np.hstack((data, np.ones(shape= (data.shape[0], 1)) * class_num))
    return data

class StatisticStack:
    '''
    计算一系列加速度帧统计、时域、频域特征，并组合为特征向量的转置
    '''
    __slots__ = ('__time_series', '__mean', '__var', '__std', '__median', '__min', '__max',
                 '__max_min', '__interquartile_range', '__kurtosis', '__skewness', '__rms',
                 '__integral', '__mean_cross_rate', '__DC', '__spectral_energy',
                 '__fft_1', '__fft_2', '__fft_3', '__fft_4', '__fft_5', '__fft_6')

    @staticmethod
    def fft(time_series):
        '''
        对时域信号进行快速傅里叶变换
        :param time_series: 待处理的时间序列
        :return: 时间序列映射到频域后的频谱信号
        '''
        return sci.fft(time_series)

    @staticmethod
    def corr(time_series):
        '''
        计算皮尔森相关系数
        :param time_series: 待处理序列
        :return: 皮尔森相关系数计算结果
        '''
        series_1 = time_series[:-2]
        series_2 = time_series[2:]
        mean = np.mean(time_series)
        #平方期望
        e_mean2 = np.sum((time_series - mean)**2) if np.sum((time_series - mean)**2) < 0.01 else \
            np.sum((time_series - mean)**2) + 0.01
        return np.sum((series_1 - mean) * (series_2 - mean)) / e_mean2


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
        self.__integral = np.sum(self.__time_series)
        self.__mean_cross_rate = np.sum(np.where(self.__time_series > self.__mean, 1, 0)) / \
            np.sum(self.__time_series)
        self.__spectral_energy = np.sum(np.abs(StatisticStack.fft(self.__time_series))**2)
        self.__fft_1, self.__fft_2, self.__fft_3, self.__fft_4, self.__fft_5, self.__fft_6 = \
            np.abs(StatisticStack.fft(self.__time_series)[1:7])

    def feature_stack(self):
        '''
        对每个序列的所有特征组合为特征向量
        :return: 数据集中单个列按采样频率计算滑动窗口计算后的特征向量, shape= (20,)
        '''
        return np.array([self.__mean, self.__var, self.__std, self.__median, self.__min, self.__max, self.__max_min,
                         self.__interquartile_range, self.__kurtosis, self.__skewness, self.__integral,self.__mean_cross_rate,
                         self.__spectral_energy, self.__fft_1, self.__fft_2, self.__fft_3, self.__fft_4,
                         self.__fft_5, self.__fft_6])

def GravityEstimate(origin, series, series_finally):
    '''
    估计重力加速度的数值
    :param origin: 初始值(tuple),第一次为(Gest= np.array([-2.3, 3.9, 8.8]), VarIncrease= 0.35, THvar= 1.0)，移动窗过程中每次都使用上一次窗口内最终得到的三值列表
    :param series: 窗口内序列，shape=(100, 3)
    :param series_finally: 扩大窗口长度为原来的4倍后的窗口内数据，shape=(400, 3)
    :return: 更新后的origin元组，其中Gest= (gx, gy, gz)
    '''
    #不变量
    SubNorm = 4
    OriginTHvar = 1.0
    hardThreshold = 1.5
    inc = 0.3
    #origin解析
    Gest, VarIncrease, THvar = origin
    #计算窗内加速度合矢量均值和方差
    W_mean = np.mean(series, axis= 0) #shape=(3,)
    #计算窗口内平均值的模值,重力加速度矢量模值
    W_mean_model = np.sqrt(np.sum(W_mean ** 2))
    Gest_model = np.sqrt(np.sum(Gest**2))
    # 计算每帧加速度矢量和
    acc_model = np.sqrt(np.sum(series**2, axis= 1)) #shape= (100,)
    W_var = np.var(acc_model)
    #如果滑动窗口内的加速度均值和估计的重力加速度相差较大，则复位方差阈值
    if np.abs(W_mean_model - Gest_model) >= SubNorm:
        THvar = OriginTHvar
    if W_var < hardThreshold:
        if W_var < THvar:
            Gest = W_mean #shape= (3,)
            THvar = (W_var + THvar) / 2
            VarIncrease = THvar * inc
        else:
            THvar = THvar + VarIncrease
    else:
        acc_finally = np.mean(series_finally, axis= 0) #shape= (3,),type= 'ndarray'
        Gest = acc_finally
    origin_new = [Gest, VarIncrease, THvar]
    return tuple(origin_new) #其中Gest=(gx, gy, gz)

def Acc_h(a, g):
    '''
    计算出去除重力加速度和在重力加速度方向上分量后的水平加速度
    :param a: 加速度计测量的三维加速度坐标(ax, ay, az)
    :param g: 估算出来的三维重力加速度g
    :return: ndarray, (h_magnitude, p)
    '''
    #去除重力加速度后的加速度
    d = a - g #a.shape=(100, 3), g.shape=(3,)
    # print(g.shape)
    #d在重力加速度方向上的分量
    p = np.matmul(d, g[:, np.newaxis]) / np.matmul(g, g[:, np.newaxis]) #shape= (100, 1)
    #d在水平方向上的分量
    h = d - (p * g[np.newaxis, :]) #shape= (100,1)*(1,3) = (100, 3)
    h_magnitude = np.sqrt(np.sum(h**2, axis= 1)) #shape= (100,)
    h_p = np.hstack((h_magnitude[:, np.newaxis], p))  # shape= (100, 2)
    return h_p[:50, :]

def matrix_operation(data):
    '''
    对进行处理后的数据集进行滑动窗口特征计算，并生成数据矩阵
    :param data: 待处理数据,shape= (62550, 9+1)
    :return: 数据矩阵
    '''
    dataset = np.zeros(shape= (1, 9*19)) #此处9*19需要修改为9*每列具有的所有特征数（统计+时域+频域）总和

    for i in range(0, data.shape[0]-50, 50): #防止越界
        feature_dataset = np.zeros(shape=(1, 20))  # 此处20需要修改为每列具有的所有特征数（统计+时域+频域）总和
        #因为data最后一列为标签
        for j in range(data.shape[-1]-1):
            statisticstack = StatisticStack(data[i:i+100, j])
            feature_stack = statisticstack.feature_stack()[np.newaxis, :]
            feature_dataset = feature_stack if feature_dataset.any() == 0 else \
                np.hstack((feature_dataset, feature_stack))

        dataset = feature_dataset if dataset.any() == 0 else np.vstack((dataset, feature_dataset))

    #将特征矩阵和标签向量进行组合并返回
    # print(dataset.shape, data.shape)
    return np.hstack((dataset, data[:dataset.shape[0], -1][:, np.newaxis]))

def data_main(path, num):
    '''
    数据处理主函数
    :param path: 其中一个交通模式数据集的绝对路径
    :param num: 需要处理的类别编号
    :return: None
    '''
    #导入一种交通模式经过去噪、均衡化后的数据集
    pri_data = LoadFile(path)
    #切出前三列进行去除重力
    pri_acc = pri_data[:, :3]
    # 初始化参数元组(在每一次窗口滑动到新位置时会迭代更新)
    origin = tuple([np.array([-2.3, 3.9, 8.8]), 0.35, 1.0])
    #初始化水平、竖直加速度分解矩阵
    h_p = np.zeros(shape= (100, 2))
    #每个窗口内的每帧三个维度加速度分量分别去除重力、分解水平、竖直
    for i in range(0, pri_acc.shape[0]-50, 50): #防止越界
        #切出shape=(100, 3)矩阵
        series = pri_acc[i:i+100, :]
        #切出4倍窗长度矩阵
        if i <= (pri_acc.shape[0] - 200):
            interval_former = (i - 200) if i >= 200 else 0
            interval_latter = interval_former + 400
        else: #数据集必须大于400个！
            interval_latter = pri_acc.shape[0]
            interval_former = interval_latter - 400
        series_finally = pri_acc[interval_former:interval_latter, :]
        origin = GravityEstimate(origin= origin, series= series, series_finally= series_finally)
        # 分解加速度为水平、竖直分量
        h_p = Acc_h(series, origin[0]) if h_p.any() == 0 else np.vstack((h_p, Acc_h(series, origin[0])))
    # print(h_p.shape)
    data = np.hstack((h_p, pri_data[:62550, 3:]))  # 组合替换加速度,去掉pri_data后50个数据
    # print(data)
    # 特征提取、组合标签得到最终待训练数据集
    data_finally = matrix_operation(data)
    SaveFile(data= data_finally, savepickle_p= r'F:\GraduateDesigning\c_%s_finallydata.pickle' % num)
    # print(data_finally)
    # print(data_finally.shape)


if __name__ == '__main__':
    #生成均衡和去噪后数据
    # for i in range(1, 7):
    #     data = EquilibriumDenoising(p_former=r'F:\GraduateDesigning', class_num=i)
    #     # dataframe = pd.DataFrame(data=data, index=list(range(1, 62551)),
    #     #                          columns=['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z',
    #     #                                   'mag_x', 'mag_y', 'mag_z', 'pre', 'mode_num'])
    #     # print(dataframe)
    #     # print(data.shape)
    #     SaveFile(data, savepickle_p=r'F:\GraduateDesigning\c_%s.pickle' % i)
    # for i in range(1, 7):
    #     data_main(path= r'F:\GraduateDesigning\c_%s.pickle' % i, num= i)
    #组合6组和后四组数据得到最后数据集
    # data_all = np.zeros(shape= (1250, 181))
    # for i in range(3, 7):
    #     data = LoadFile(p= r'F:\GraduateDesigning\c_%s_finallydata.pickle' % i)
    #     # print(data[0, 0])
    #     data_all = data if data_all.any() == 0 else np.vstack((data_all, data))
    # SaveFile(data= data_all, savepickle_p= r'F:\GraduateDesigning\data_sim.pickle')

    #检查缺失值
    data = LoadFile(p= r'F:\GraduateDesigning\data_all.pickle')
    # print(data[:, 171])
    nan = []
    for i in range(data.shape[-1]):
        if np.isnan(data[:, i]).any():
            nan.append(i)
    print(len(nan), nan)







