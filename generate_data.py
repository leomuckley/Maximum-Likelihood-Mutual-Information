import numpy as np
import matplotlib.pyplot as plt
from synthetic_data import Gaussian



class MutualInfo():
    
    def __init__(self, dim, corr=(0,1), cond=(10,10), nsamples=500, seed=None):
        self.gauss = Gaussian(seed)
        self.dim = dim
        self.corr = corr
        self.cond = cond
        self.nsamples = nsamples
        
    def true_mutual_info(self):
        data_x, data_y, MI, params = self.gauss.generate_example(dim=self.dim, 
                                                                 nsamples=self.nsamples, 
                                                                 rhoz_lims=(self.corr[0], self.corr[1]), 
                                                                 cond_numbers=(self.cond[0], self.cond[1]))
        return data_x, data_y, MI, params
        
        
    
    def plot(self):
        
        # Clear Fig.
        plt.clf()
        x, y, mi, params = self.true_mutual_info()
        plt.scatter(x, y, marker='o', norm=0.5)
        plt.title('True MI = {}'.format(np.round(mi, 4)))
        plt.xlabel('x')
        plt.ylabel('y')
        
        return plt.show()
    
    


