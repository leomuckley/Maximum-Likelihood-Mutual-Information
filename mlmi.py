#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 09:15:27 2019

@author: leo
"""

import numpy as np

def MLMI(x, y, y_type=0, sigma_list=np.logspace(-2, 2, 9), b=200, fold=5):
    
    my_tol = 10e-15
    
    n = x.shape[1]
    ny = y.shape[1]
    
    if n != ny:
        print("Number of smaples of x and y must be the same!!!")
    
    # Have at most 200 Gaussian centres    
    b = np.min(n, b)
    
    # Gaussian centres are randomly chosen from samples
    rand_index = np.random_permutation(n)
    u = x[:, rand_index[0:b-1]]
    v = y[:, rand_index[0:b-1]]
    
    dy = y.shape[0]
    
"""    
    Phi_tmp = GaussBasis_sub()
""" 