import numpy as np
import matplotlib.pyplot as plt
from synthetic_data import Gaussian



class MutualInfo():
    """An object for generating true mutual information data.
    
    Parameters
    ----------
    dim       : int, default=1
                Specify the dimension of the problem.
    corr      : tuple, default=(0,1)
                These values control the value of the mutual information that
                is generated. The default values (0,1), generate random vectors
                with mutual information values in this range. For example, to 
                generate random vectors with larger mutual information values,
                set the lower limit closer to one (e.g. 0.9).
    n_samples : int, default=500
                The number samples to generate from the random vectors.
    seed      : int, default=None
                To control the random state, this optional argument can be
                utilised to replicate results.     
    """
    
    
    def __init__(self, dim=1, corr=(0,1), cond=(10,10), n_samples=500, seed=None):
        self.gauss = Gaussian(seed)
        self.dim = dim
        self.corr = corr
        self.cond = cond
        self.n_samples = n_samples
        
        
    def true_mutual_info(self):
        data_x, data_y, MI, params = self.gauss.generate_example(dim=self.dim, 
                                                                 n_samples=self.n_samples, 
                                                                 rhoz_lims=(self.corr[0], self.corr[1]), 
                                                                 cond=(self.cond[0], self.cond[1]))
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
    
    


