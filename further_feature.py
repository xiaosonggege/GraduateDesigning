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

rng = ra.RandomState(0)
a = rng.random_integers(low= 1, high= 3, size= (1000,))
for i in range(100, 900, 200):
    a[i:i+10] = rng.random_integers(low= 10, high= 16, size= (10,))
    if not (i % 20):
        a = -1 * a

# plt.figure('加速度图')
# plt.plot(a)
# plt.show()


