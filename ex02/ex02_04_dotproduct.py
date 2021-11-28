# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

def init(size):
    x = []
    y = []
    for i in range(size):
        x.append(float(i))
        y.append(float(size - i - 1))
    return np.array(x),np.array(y)


def tester(size):
    x,y = init(size)
#    print("x=")
#    print(x)
#    print("y=")
#    print(y)
#    
#    print("size = {}, np.dot(x,y) = ".format(size))
    print(np.dot(x,y))
    print("{:1.5e}".format(np.dot(x,y)))


if __name__ == "__main__":
    print("blub")