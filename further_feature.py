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
import pickle
from scipy import stats
from data_operation import LoadFile, SaveFile

#峰特征类
class Peak:
    '''峰特征提取类'''

    __slots__ = ('__scan_series', '__window_length', '__start_Th', '__end_Th',
                 '__peak_area_formmer', '__segfeature', '__peakfeatureformmer')

    @staticmethod
    def calc_feature(peak_area):
        '''
        对峰值区域计算相关特征
        :param peak_area: 待处理峰值区域
        :return: 峰值特征向量
        '''
        #积分
        Auc = np.trapz(y= peak_area, dx= 0.01)
        #长度
        length = peak_area.shape[0] / 100
        #强度
        intensity = np.mean(peak_area)
        #kurtosis
        kurtosis = stats.kurtosis(peak_area)
        #skewness
        skewness = stats.skew(peak_area)
        return np.array([Auc, length, intensity, kurtosis, skewness])

    def __init__(self, scan_series, window_length, start_Th, end_Th, peak_area_formmer):
        '''
        峰特征类构造函数
        :param scan_series: 待处理序列（长度为400）
        :param window_length: 相邻滑动窗口长度(一半为10)
        :param start_Th: 峰起阈值（左右窗口中平均值）
        :param end_Th: 峰落阈值（左右窗口中平均值）
        :param peak_area_formmer: 前一窗口峰值区域遗留部分
        '''
        self.__scan_series = scan_series
        self.__window_length = window_length
        self.__start_Th = start_Th
        self.__end_Th = end_Th
        self.__peak_area_formmer = peak_area_formmer
        self.__segfeature = np.zeros(shape= (1, 5)) #用于存储峰特征，indexs: peakfeature, columns: feature
        self.__peakfeatureformmer = [] #用于存储起始点在前一个窗口中的峰值区域

    def find_peak_areas(self):
        '''
        找到序列中所有峰值区域
        :return: type= tuple, 所有峰值区域起始点作为元组元素输出和峰值区域未结束的前半部分
        '''
        #峰值区域起始点存储列表
        region = []
        #设置峰值区域起始、结束点标记
        start_point, end_point = None, None
        #设置前半窗口和后半窗口均值
        pre_mean, late_mean = 0, 0
        #设置标记变量
        prefix_left = 0
        if len(self.__peak_area_formmer): #如果上一次有遗留的峰值区域，则prefix_left置1
            prefix_left = 1

        for position in range(self.__window_length, len(self.__scan_series)-self.__window_length): #为相邻滑动窗口留出空间
            #计算相邻窗口各自均值
            pre_mean = np.mean(self.__scan_series[position-self.__window_length: position])
            late_mean = np.mean(self.__scan_series[position: position+self.__window_length])
            if prefix_left:
                start_point = -1
            elif pre_mean < self.__start_Th and late_mean > self.__start_Th:
                start_point = position

            if start_point != None or prefix_left: #确保峰值区域起始点已经找到
                if pre_mean > self.__end_Th and late_mean < self.__start_Th :
                    end_point = position
                    region.append((start_point, end_point)) #注意region第一个元组元素的start_point可能为-1
                    start_point, end_point = None, None
                    if prefix_left:
                        prefix_left = 0

        # 判断峰值区域是否在该窗口中结束
        if start_point != None and end_point == None:
            sub_peak_area = self.__scan_series[start_point:] if not prefix_left else \
                np.hstack((self.__peak_area_formmer.pop(), self.__scan_series.copy()))
            self.__peakfeatureformmer.append(sub_peak_area)

        return tuple(region), self.__peakfeatureformmer

    def PeakFeatureExtract(self, region, peakfeatureformmer):
        '''
        提取峰值特征
        :param region:  存储所有峰值区域起始点作为元素的元组
        :param peakfeatureformmer: 存储峰值区域未结束的前半部分的列表, 可以为空
        :return: 峰值特征向量、峰特征存储矩阵（用于计算段特征）
        '''
        #初始化峰特征向量均值
        peakfeature_mean = np.zeros(shape= (1, 5))
        echo = 0
        for interval in region:
            if interval[0] == -1:
                series = np.hstack((peakfeatureformmer, self.__scan_series[: interval[-1]]))
            else:
                series = self.__scan_series[interval[0]: interval[-1]]
            peakfeature = Peak.calc_feature(peak_area= series)
            self.__segfeature = np.vstack((self.__segfeature, peakfeature))
            peakfeature_mean = (peakfeature_mean * echo + peakfeature) / (echo + 1)
        return peakfeature_mean, self.__segfeature

def peak_main(p):
    '''
    峰特征、段特征提取主函数
    :return: None
    '''
    #导入数据(只导入一种交通模式数据)
    dataset = LoadFile(p= p)
    #取重力加速度水平和竖直分量
    dataset_acc = dataset[:, [0, 1]] #shape= (500200, 2)
    #非重叠滑动窗口，窗口长度为: 1200, 滑动距离为: 1200

        # #创建峰特征类对象
        # peak = Peak(scan_series= , window_length= 1200, start_Th= 0.2, end_Th= 0.2, peak_area_formmer= )















