#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 15:18:31 2019

@author: leo
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import matlib
from generate_data import MutualInfo




class MLMI():
    """An object for Maximum-Likelihood Mutual Information estimation.
    
    Parameters
    ----------
    x        :  array_like, shape=(n_samples, n_dims)
                X-vector.
    y        :  array_like, shape=(n_samples, n_dims)
                Y-vector
    """
    
    def __init__(self, x, y):
        self.x = x.reshape(1, len(x.T))
        self.y = y.reshape(1, len(y.T))
        
        
    def GaussBasis_sub(self, z, c):
        """
        Gaussian Kernel model: for z = (x', y')' , and choose Gaussian centres c
        randomly from the set z. 
        The number of basis functions at b=min(200, n) and choose the Gaussian
        width, sigma, by cross validation.
        
        Parameters
        ----------
        z        : array_like, (x', y')'
        c        : array_like, Gaussian Centres
        
        Returns
        -------
        Phi_temp : array_like, 
                   Temporary Phi
        
        """
        # Calculating everything except the sigma
        Phi_temp = - (           
                matlib.repmat(np.sum(c**2, axis=0, keepdims=True),
                              z.shape[1],
                              1) +
                matlib.repmat(np.sum(z**2, axis=0, keepdims=True).T,
                              1,
                              c.shape[1]) - 2.0 * z.T * c) / 2.0
                
        return Phi_temp
    
    
    def GaussBasis(self, Phi_temp, sigma):
        """ Mehtod to imlement the Gaussian basis functions. 
        
        Parameters
        ----------
        z        : array_like, (x', y')'
        c        : array_like, Gaussian Centres
        
        Returns
        -------
        Phi_temp  : array_like, Full Phi
        
        """
        
        Phi = np.exp(Phi_temp / (sigma)**2)
        
        return Phi
        
         
    
    def KLIEP_projection(self, alpha, Xte, b, c):    
        
        mytol = 10e-15
        #infinity = math.inf
       
        alpha_zero = b * (1 - np.matmul(b.T, alpha)) / c
        #alpha_tmp = alpha_tmp.reshape(len(alpha_tmp), 1)
        alpha = np.add(alpha, alpha_zero) # (b, 1)
        alpha = np.maximum(0, alpha)    
        ww = np.matmul(b.T, alpha)    
                
        ww[np.where(np.abs(ww) < 10e-20)] = np.inf    
        alpha = alpha/ww
        Xte_alpha = np.matmul(Xte, alpha) # (500, 1)
        score = np.mean(np.log(Xte_alpha[np.where(Xte_alpha > mytol)]))
        
        return alpha, Xte_alpha, score
        
          
    
    def KLIEP_learning(self, b, Xte):
        
        mytol = 1e-15
            
        alpha = np.ones((Xte.shape[1], 1)) # (b, 1)
        
        #ntr = len(b)
        nte, nc = Xte.shape
        
        max_iteration = 100
        epsilon_list = [10**i for i in range(3, -4, -1)] # length i    
        c = np.sum(b**2, axis=0, keepdims=True)
        
        alpha, Xte_alpha, score = self.KLIEP_projection(alpha, Xte, b, c)
        # alpha, Xte is (500, 500)
        # Score is a float    
        #XXte = Xte*Xte.T    
        #store_eye = np.eye(nc)
        
        for epsilon in epsilon_list:
            for iteration in range(max_iteration):
                inwxte = 1 / Xte_alpha
                inwxte[np.where(Xte_alpha < mytol)] = 0
                ww = np.matmul(Xte.T, inwxte)
                epsilon_scale = epsilon
                alpha_tmp = alpha + epsilon_scale * ww # (200, 1)
                alpha_new, Xte_alpha_new, score_new = self.KLIEP_projection(alpha_tmp, Xte, b, c)
                
                if (score_new - score) <= 0:
                    break
                
                score = score_new
                alpha = alpha_new
                Xte_alpha = Xte_alpha_new
                
        return alpha, score
    
    
    def fit(self):
        """ Apply MLMI to data to estimate mutual information. """
        mytol = 1e-15
        # The length of the x-vector
        n = self.x.shape[1]
        
        b = 200  # b is the number of kernel centres
        b = np.minimum(b, n)       
        
        #Gaussian centers are randomly chosen from samples        
        rand_index = np.random.permutation(n) # (n,)
        
        u = self.x[:, rand_index[:b]] # (1, b)
        v = self.y[:, rand_index[:b]] # (1, b)  
        sigma = 0.31623

        phix_tmp = self.GaussBasis_sub(self.x, u).T # (b, n)
        phiy_tmp = self.GaussBasis_sub(self.y, v).T  
        phi_x = self.GaussBasis(phix_tmp, sigma)
        phi_y = self.GaussBasis(phiy_tmp, sigma)                  
         
        Phi = phi_x * phi_y # (b, n)        
        # bb is the constraint and the output is (n, n)
        bb = (np.sum(phi_x, axis=1, keepdims=True) * np.sum(phi_y, axis=1, keepdims=True) - 
             np.sum(phi_x * phi_y, axis=1, keepdims=True)) /  (n * (n-1))  # (b, 1)            
        
        alpha_hat, score = self.KLIEP_learning(bb, Phi.T)        
        w_hat = np.matmul(alpha_hat.T, Phi) # (1, N)        
        MI_hat = np.mean(np.log(w_hat[np.where(w_hat > mytol)]))        
        
        # Return predicted mutual information
        return MI_hat
    
        
    def predict(self, n_iter=100):
        """ Average n predictions to return final MI score. """
        res_list = []
        for _ in range(n_iter):
            res_list.append(np.abs(self.fit()))
        return np.mean(res_list)
    
   
    
   
    
   
    
if __name__ == '__main__':
    SEED = 111
    err_list = []
    for i in range(10, 1010, 10):
        mi = MutualInfo(dim=1, corr=(0.8,1.0), cond=(10,10), n_samples=i, seed=SEED)
        data_x, data_y, MI, params = mi.true_mutual_info()
        mlmi = MLMI(data_x, data_y)
        MI_hat = np.round(mlmi.predict(), 4)
        err = np.round(np.abs(MI - MI_hat), 2)
        err_list.append((i,err))
        #print(f"The MLMI model predicted {np.round(mlmi.predict(), 4)} based the data")
    plt.plot(*zip(*err_list))
    plt.title("Mutual Information Estimation Error")
    plt.xlabel("Number of samples")
    plt.ylabel("Mean estimation error")
    plt.show()

    
    