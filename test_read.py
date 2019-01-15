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
    a = np.arange(20).reshape(4, 5)
    def fun(a):
        b = a
        np.random.shuffle(b)
        return b
    c = fun(a)
    print(c)






