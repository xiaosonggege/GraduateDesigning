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

import multiprocessing
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# def run(lock, q, num):
#     with lock:
#         print('run')
#         for i in range(num):
#             q.put(i, block=False)
#
#
# def run_1(lock, q, num):
#     lock.acquire()
#     print('run_1')
#     try:
#         for i in range(num):
#             print(q.get(block=False))
#     except:
#         pass
#     finally:
#         lock.release()
#
# def run_2(lock, q, num):
#     lock.acquire()
#     print('run_2')
#     try:
#         for i in range(num):
#             q.put(i, block= False)
#     except:
#         pass
#     finally:
#         lock.release()

def fun(num):
    print(num)


if __name__ == '__main__':
    # q = multiprocessing.Queue()
    # lock = multiprocessing.Lock()
    # p = multiprocessing.Process(target= run, args= (lock, q, 6,))
    # p_1 = multiprocessing.Process(target= run_1, args= (lock, q, 12,))
    # p_2 = multiprocessing.Process(target= run_2, args= (lock, q, 6,))
    # p.start()
    # p_2.start()
    # p_1.start()
    # p_1.join()
    # print(q)
    # pool = multiprocessing.Pool(processes= 4)
    # for i in range(4):
    #     pool.apply(fun, args= (i, ))
    #
    # pool.close()
    # pool.join()
    # a = np.arange(20).reshape(4, 5)
    # b = np.arange(30).reshape(5, 6)
    # p = r'C:\Users\xiaosong\Desktop\ttt.pickle'
    # with open(p, 'wb') as file:
    #     print(file.tell())
    #     file.seek(0,1)
    #     pickle.dump(a, file, -1)
    #     print(file.tell())
    # with open(p, 'ab') as file:
    #     print(file.tell())
    #     file.seek(0)
    #     print(file.tell())
    #     pickle.dump(b, file, -1)
    # with open(p, 'ab+') as file:
    #     file.seek(0)
    #     a = pickle.load(file)
    #     print(file.tell())
    #     print(a)
    #     file.seek(231)
    #     b = pickle.load(file)
    #     print(file.tell())
    #     print(b)
    # a = np.arange(200).reshape(20, 10)
    # # a = np.delete(a, [0, 1, 2, 3, 4], axis= 1)
    # data = pd.DataFrame(data= a, columns= ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z',
    #                                        'mag_x', 'mag_y', 'mag_z', 'pre'], index= list(range(20)))
    # print(data)
    # # data.loc()
    # data = data[(~data['acc_x'].isin([190])) & (~data['acc_y'].isin([191]))]
    # data = data.loc[data['pre'] > 190]
    # print(data)
    a_pre = np.array([1, 2, 3, 4, 1, 6, 1, 4, 3, 2, 1])
    b_fact = np.array([2, 2, 3, 4, 5, 1, 5, 4, 3, 1, 1])
    recall = np.zeros(shape= (6, 2))
    pre = np.zeros(shape= (6, 2))
    for i in range(1, 7):
        pre_bool = np.where(a_pre == i, 1, 0)
        fact_bool = np.where(b_fact == i, 1, 0)
        #计算预测值为i的总数
        sum_pre = np.sum(pre_bool)
        #计算实际值为i的总数
        sum_fact = np.sum(fact_bool)
        pre_fact = (pre_bool & fact_bool)
        #计算实际值和预测值一致的总数
        sum_con = np.sum(pre_fact)
        #将实际值部分存入b_fact
        recall[i-1] = np.array([sum_con, sum_fact - sum_con])
        pre[i-1] = np.array([sum_con, sum_pre - sum_con])
        #将预测值部分存入a_pre
    pre_pd = pd.DataFrame(data= pre, index= [i for i in range(1, 7)], columns= ['TP', 'FP'])
    pre_pd.eval('precision_rate = TP / (TP + FP)', inplace= True)
    recall_pd = pd.DataFrame(data= recall, index= [i for i in range(1, 7)], columns= ['TP', 'FN'])
    recall_pd.eval('recall_rate = TP / (TP + FN)' , inplace= True)
    print(pre_pd)
    print(recall_pd)
    # recall_pd['recall_rate'].apply(lambda x: x.astype(np.float64))
    recall_rate_ave = np.mean(recall_pd['recall_rate'])
    precision_rate_ave = np.mean(pre_pd['precision_rate'])
    F1_score = 2*precision_rate_ave*recall_rate_ave / (precision_rate_ave + recall_rate_ave)
    print(precision_rate_ave)
    print(recall_rate_ave)
    print(F1_score)











