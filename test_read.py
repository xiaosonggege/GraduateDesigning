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
    q = multiprocessing.Queue()
    lock = multiprocessing.Lock()
    # p = multiprocessing.Process(target= run, args= (lock, q, 6,))
    # p_1 = multiprocessing.Process(target= run_1, args= (lock, q, 12,))
    # p_2 = multiprocessing.Process(target= run_2, args= (lock, q, 6,))
    # p.start()
    # p_2.start()
    # p_1.start()
    # p_1.join()
    # print(q)
    pool = multiprocessing.Pool(processes= 4)
    for i in range(4):
        pool.apply(fun, args= (i, ))

    pool.close()
    pool.join()



