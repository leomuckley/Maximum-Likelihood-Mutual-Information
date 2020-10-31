#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 15:01:20 2020

@author: leo
"""

from synthetic_data import Gaussian
from mlmi import MLMI
from generate_data import MutualInfo
import matplotlib.pyplot as plt
import numpy as np

SEED = 123

err_list = []
for i in range(10, 1010, 10):
        mi = MutualInfo(dim=1, corr=(0.8,1.0), cond=(10,10), n_samples=i, seed=SEED)
        data_x, data_y, MI, params = mi.true_mutual_info()
        mlmi = MLMI(data_x, data_y)
        MI_hat = np.round(mlmi.predict(), 4)
        err = np.round(np.abs(MI - MI_hat), 2)
        err_list.append((i,err))

# Plot approximation error
plt.plot(*zip(*err_list))
plt.title("Mutual Information Estimation Error")
plt.xlabel("Number of samples")
plt.ylabel("Mean estimation error")
plt.show()