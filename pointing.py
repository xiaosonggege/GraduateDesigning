#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2018-2021, Node Supply Chain Manager Corporation Limited.
@contact: 1243049371@qq.com
@software:PyCharm
@file: pointing
@time: 2019/1/8 22:46
@desc:
'''

import numpy as np
from matplotlib import pyplot as plt
import pandas

index = np.arange(5)
bar_width = 0.35
opacity = 0.4
means_men = (20, 35, 30, 35, 27)
figure_1 = plt.figure()
plt.bar(index, means_men, bar_width, alpha=opacity,
        color= ['r', 'g', 'b'], label= 'Men')
plt.xlabel('group')
plt.ylabel('scores')
plt.xticks(index , ('A', 'B', 'C', 'D', 'E'))
plt.legend()
plt.grid(linestyle= '--')
plt.ylim(0, 60)
for a, b in zip(index-0.05, means_men):
    plt.text(a, b+1, '%.0f' % b)
plt.show()