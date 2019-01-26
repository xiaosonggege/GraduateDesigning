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
        :param scan_series: 待处理序列
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
        # print('传入的region中元组第一个索引是: %s, peakfeatureformmer长度为: %s\n' % (region[0][0], len(peakfeatureformmer)))
        #初始化峰特征向量均值
        peakfeature_mean = np.zeros(shape= (1, 5))
        echo = 0
        for interval in region:
            if interval[0] == -1:
                # if len(peakfeatureformmer):
                #     formmer = peakfeatureformmer.pop()
                # else:
                #     formmer = np.zeros(shape= (1, 500))
                # latter = self.__scan_series[:interval[-1]]
                # print(formmer.shape, latter[np.newaxis, :].shape)
                # series = np.hstack((formmer[np.newaxis, :], latter[np.newaxis, :]))
                series = self.__scan_series[:interval[-1]]
            else:
                series = self.__scan_series[interval[0]: interval[-1]]
            peakfeature = Peak.calc_feature(peak_area= series)
            self.__segfeature = np.vstack((self.__segfeature, peakfeature))
            peakfeature_mean = (peakfeature_mean * echo + peakfeature) / (echo + 1)
        return peakfeature_mean, self.__segfeature

    def segcalc(self, segfeature):
        '''
        计算段特征，判断如果存储峰特征的矩阵行数大于3时计算，否则为相同尺度的0向量
        :param segfeature: 峰特征矩阵
        :return: 当前窗口内的段特征
        '''
        if segfeature.shape[0] >= 3:
            return np.var(segfeature, axis= 0)
        else:
            return np.zeros(shape= (1, 5))

def peak_main(p, num):
    '''
    峰特征、段特征提取主函数
    :param p: 读取原始数据路径
    :param num: 待保存文件编号
    :return: None
    '''
    #初始化遗留峰值区域向量
    peak_featureformmer = []
    #导入数据(只导入一种交通模式数据)
    dataset = LoadFile(p= p)
    #取重力加速度水平和竖直分量
    dataset_acc = dataset[:, [0, 1]] #shape= (500200, 2)

    #峰特征初始化（没有捕捉到峰值区域时峰特征对应位置均为0）
    peak_feature_h, peak_feature_v = np.zeros(shape= (1, 5)), np.zeros(shape= (1, 5))

    #初始化段特征向量（没有得到段特征的对应位置均为0），两倍峰特征提取时间的风特征矩阵初始化、以及段特征计算标志
    seg_feature_h, seg_feature_v, twice_segfeature, seg_yes = \
        np.zeros(shape= (1, 5)), np.zeros(shape= (1, 5)), np.zeros(shape= (1, 5)), 1

    #非重叠滑动窗口，窗口长度为: 1200, 滑动距离为: 1200,水平和竖直分量都需要提取
    for column_num in range(dataset_acc.shape[-1]):
        for window_position in range(0, dataset_acc.shape[0] - 1000 + 1, 1000):
            peak_area_formmer = peak_featureformmer
            # 创建峰特征类对象
            peak = Peak(scan_series=dataset_acc[window_position:window_position + 1000, column_num], window_length=10,
                        start_Th=0.2, end_Th=0.2, peak_area_formmer=peak_area_formmer)
            # 得到峰值区域位置元组和未完峰值区域
            region, peak_featureformmer = peak.find_peak_areas()
            # print('接收到的region长度为: %s, peak_featureformmer长度为: %s' % (len(region), len(peak_featureformmer)))
            # 得到单列峰特征平均值和单列段特征矩阵
            peakfeature_mean, segfeature = peak.PeakFeatureExtract(region=region, peakfeatureformmer=peak_featureformmer)
            # print(peakfeature_mean)
            #5组峰特征为组成子特征矩阵
            sub_peakfeature = np.vstack((np.zeros(shape= (4, 5)), peakfeature_mean))
            if not column_num:
                peak_feature_h = sub_peakfeature if not window_position else np.vstack((peak_feature_h, sub_peakfeature))
                # print('峰特征为 %s, %s' % peak_feature_h.shape)
            else:
                peak_feature_v = sub_peakfeature if not window_position else np.vstack((peak_feature_v, sub_peakfeature))

            #得到单列段特征向量(每得到两组相邻峰特征矩阵后计算一次段特征)
            if seg_yes == 1:
                twice_segfeature = segfeature
                seg_yes += 1
            else:
                twice_segfeature = np.vstack((twice_segfeature, segfeature))
                sub_seg_feature = peak.segcalc(segfeature= twice_segfeature)
                sub_seg_feature = np.vstack((np.zeros(shape= (9, 5)), sub_seg_feature))
                if not column_num:
                    seg_feature_h = sub_seg_feature if window_position == 1000 else np.vstack(
                        (seg_feature_h, sub_seg_feature))
                    # print('段特征为: %s, %s' % seg_feature_h.shape)
                else:
                    seg_feature_v = sub_seg_feature if window_position == 1000 else np.vstack(
                        (seg_feature_v, sub_seg_feature))

                #计算段特征标志置1
                seg_yes = 1
                #twice_segfeature矩阵清空
                twice_segfeature = np.zeros(shape= (1, 5))


    #得到组合后的峰特征shape=（500200， 10）
    peak_feature = np.hstack((peak_feature_h, peak_feature_v))
    # print(peak_feature.shape)

    #得到组合后的段特征shape=（500200， 10）
    seg_feature = np.hstack((seg_feature_h, seg_feature_v))
    # print(seg_feature.shape)

    #得到峰特征和段特征的组合shape=（500200， 20）
    peak_seg = np.hstack((peak_feature, seg_feature))
    SaveFile(peak_seg, savepickle_p= r'F:\GraduateDesigning\PeakSegdataset\c_peakseg_%s.pickle' % num)
    print(peak_seg.shape)

if __name__ == '__main__':
    # for num in range(3, 7):
    #     p = r'F:\GraduateDesigning\PreoperationData\c_preop_%s.pickle' % num
    #     peak_main(p, num= num)
    # data = LoadFile(p= r'F:\GraduateDesigning\PeakSegdataset\c_peakseg_3.pickle')
    # print(data == np.zeros(shape= data.shape))
    #合并4组峰段特征
    # dataset_peak_seg = np.zeros(shape= (1, 20))
    # for i in range(3, 7):
    #     data = LoadFile(p= r'F:\GraduateDesigning\PeakSegdataset\c_peakseg_%s.pickle' % i)
    #     dataset_peak_seg = data if dataset_peak_seg.any() == 0 else np.vstack((dataset_peak_seg, data))
    # print(dataset_peak_seg.shape)
    # SaveFile(data= dataset_peak_seg, savepickle_p= r'F:\GraduateDesigning\finalDataset\data_peak_sim.pickle')
    #组合帧特征和峰特征和段特征
    data_frame = LoadFile(p= r'F:\GraduateDesigning\finalDataset\data_frame_sim.pickle')
    data_peak_seg = LoadFile(p= r'F:\GraduateDesigning\finalDataset\data_peak_sim.pickle')
    dataset_sim = np.hstack((data_frame[:, :-1], data_peak_seg, data_frame[:, -1][:, np.newaxis]))
    SaveFile(data= dataset_sim, savepickle_p= r'F:\GraduateDesigning\finalDataset\data_sim.pickle')
























