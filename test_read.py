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

import numpy as np
import scipy as sci
from scipy import stats
from scipy.integrate import quad
from data_operation import SaveFile
import pickle
import os

#data为4类数据经过数据均衡、去噪后的矩阵
data = np.ones(shape= [6250*4, 11])
for num in range(3, 7):
    p = r'D:\GraduateDesigning\ICT DataSet\Label_%s.txt' % num
    with open(p, 'r') as file:
        print('正在处理第%d个模型' % num)
        sub_data = np.loadtxt(file, delimiter=',', skiprows=0)[:62550, :]
        SaveFile(sub_data, r'D:\GraduateDesigning\ICT DataSet\Label_revise_%s.txt')
        # print('第%s个文件转换后的维度是: (%d, %d)' % (num, sub_data.shape[0], sub_data.shape[-1]))
#         i = 0
#         while i <= sub_data.shape[0]:
#             if sub_data[i, :].any() == 0 or sub_data[i, -2] < 0:
#                 sub_data = np.delete(sub_data, i, axis= 0)
#             else:
#                 i += 1
#                 print(i)
#         sub_data = np.delete(sub_data, [3, 4, 5], axis= 1)
#         data = sub_data[:6251, :] if data.any() == 0 else np.vstack((data, sub_data[:6251, :]))
#
