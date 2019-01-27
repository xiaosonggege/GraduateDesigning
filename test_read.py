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
delete_array_1 = [14, 17, 19]
delete_array = [14, 17, 19]
i = 1
while ((19+i*19) <= 171):
    delete_array.extend([j + i * 19 for j in delete_array_1])
    i += 1

delete_array.extend([176, 177, 178, 179, 180, 186, 187, 188, 189, 190])
print(delete_array)
print(len(delete_array))











