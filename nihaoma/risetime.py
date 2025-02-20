#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 10:58:33 2023

@author: mingzeyan
"""

import matplotlib.pyplot as plt
import numpy as np
node = np.array([0,4,8,16,20,24,28,32,40])
rise_time = np.array([756e-9,3.7884e-6,4.7712e-6,6.3784e-6,6.9552e-6,7.3668e-6,7.5852e-6,8.7556e-6,9.002e-6])
plt.plot(node,rise_time,'x')
plt.xlabel('Node')
plt.ylabel('Rise time, s')
plt.title('Rise time at different nodes')
