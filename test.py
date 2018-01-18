#!/usr/bin/env python
# _*_ coding: utf-8 _*_
"""
 @description:
        
 @Time       : 18/1/5 下午10:06
 @Author     : guomianzhuang
"""
# from __future__ import print_function
import numpy as np
import random
import math


def test():
    z = np.array([[0,0,0],[0,0,0]])
    z[0, 2] = 1
    print(z)


def test2():
    for i in range(500):
        topic = int(math.floor(random.random()*50))
        print(topic)


if __name__ == '__main__':
    # test()
    test2()