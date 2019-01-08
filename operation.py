#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software: pycharm
@file: operation
@time: 2019/1/8 12:28
@desc:
'''

import xlwt
import os.path
import numpy as np

def operation(excel):
    '''
    对矩阵进行对角线元素统计百分比
    :param excel: 待处理矩阵
    :return: 矩阵中处理后的对角线元素
    '''
    diag = np.sum(excel, axis= 1)
    excel_diag = np.diag(excel) / diag
    return excel_diag

def makeExcel(data):
    '''
    制作表格
    :param data: 待处理数据
    :return: ndarray类型矩阵对应输出excel/xls/xlsx
    '''
    file = xlwt.Workbook()
    table = file.add_sheet('statistic', cell_overwrite_ok= True)
    #建立列名
    table.write(1, 0, 'CNN')
    table.write(2, 0, 'Decision-Tree')
    table.write(3, 0, 'Random-Forest')
    table.write(4, 0, 'CNN+LSTM')
    table.write(0, 1, 'Still')
    table.write(0, 2, 'Walk')
    table.write(0, 3, 'Run')
    table.write(0, 4, 'Bike')
    table.write(0, 5, 'Car')
    table.write(0, 6, 'Bus')
    table.write(0, 7, 'Train')
    table.write(0, 8, 'Subway')
    v, h = data.shape
    for i in range(v):
        for j in range(h):
            table.write(i + 1, j + 1, str(data[i, j]))
    file.save(r'C:\Users\xiaosong\Desktop\Data_statistic.xls')



if __name__ == '__main__':
    excel_1 = np.array([
        [5773, 46, 1, 30, 7, 17, 139, 131],
        [85, 5564, 12, 24, 1, 16, 34, 58],
        [6, 42, 5376, 8, 0, 8, 1, 1],
        [57, 46, 3, 5408, 11, 24, 55, 32],
        [11, 1, 0, 4, 6351, 73, 55, 32],
        [15, 17, 1, 23, 129, 5137, 75, 114],
        [114, 14, 0, 13, 48, 48, 6022, 472],
        [262, 30, 0, 39, 220, 173, 1309, 3236]
    ])
    excel_2 = np.array([
        [2555226, 16668, 892, 24084, 33147, 26193, 55484, 50006],
        [18878, 2182901, 117367, 239279, 6338, 19642, 16342, 23083],
        [111, 11961, 2449952, 5385, 6, 51, 45, 66],
        [28308, 238571, 53835, 2079653, 29282, 31442, 25, 24],
        [190, 0, 0, 8, 5671, 439, 130, 83],
        [144, 24, 0, 78, 812, 4164, 137, 152],
        [479, 12, 0, 22, 262, 215, 4618, 1123],
        [262, 30, 0, 39, 220, 173, 1309, 3236]
    ])
    excel_3 = np.array([
        [3270077, 15788, 328, 27779, 44133, 26412, 34735, 20387],
        [14629, 2965855, 78225, 180160, 6127, 17717, 9454, 9245],
        [107, 22540, 3060960, 5347, 3, 12, 6, 24],
        [20602, 241091, 50484, 2749395, 37281, 25360, 9964, 12192],
        [36300, 2225, 55, 20607, 3398222, 130519, 81021, 46561],
        [32926, 17580, 215, 34989, 232585, 2659385, 98672, 55499],
        [79784, 11343, 161, 19850, 196819, 183146, 3120736, 163577],
        [75241, 20263, 278, 33124, 204520, 183376, 318898, 2098428]
    ])
    excel_4 = np.array([
        [6003, 35, 0, 13, 1, 5, 43, 44],
        [44, 5651, 13, 13, 2, 7, 15, 49],
        [3, 18, 5413, 0, 0, 0, 0, 0],
        [30, 40, 4, 5488, 5, 18, 36, 15],
        [5, 0, 0, 2, 6468, 30, 10, 6],
        [3, 3, 0, 9, 24, 5413, 27, 32],
        [52, 9, 0, 14, 23, 48, 6369, 216],
        [61, 45, 0, 10, 17, 70, 258, 4808]
    ])

    ex_1 = operation(excel_1)[np.newaxis, :]
    ex_2 = operation(excel_2)[np.newaxis, :]
    ex_3 = operation(excel_3)[np.newaxis, :]
    ex_4 = operation(excel_4)[np.newaxis, :]
    table = np.vstack((ex_1, ex_2, ex_3, ex_4))
    print(table)
    makeExcel(table)

