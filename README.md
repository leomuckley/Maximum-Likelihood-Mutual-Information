# A Python Package for Maximum Likelihood Mutual Information Estimation
Python code to estimate mutual information between variables using a maximum likelihood approach.

## Mutual information approximation via maximum likelihood estimation of density ratio 
---
Authors: *Taiji Suzuki* ; *Masashi Sugiyama* ; *Toshiyuki Tanaka*


### Abstract:
We propose a new method of approximating mutual information based on maximum likelihood estimation of a density ratio function.
The proposed method, Maximum Likelihood Mutual Information (MLMI), possesses useful properties, e.g., it does not involve density estimation, 
the global optimal solution can be efficiently computed, it has suitable convergence properties, 
and model selection criteria are available. 
Numerical experiments show that MLMI compares favorably with existing methods.

https://ieeexplore.ieee.org/abstract/document/5205712


**Example**

``` python
for i in range(10, 1010, 10):
        mi = MutualInfo(dim=1, corr=(0.8,1.0), cond=(10,10), nsamples=i, seed=SEED)
        data_x, data_y, MI, params = mi.true_mutual_info()
        mlmi = MLMI(data_x, data_y)
        MI_hat = np.round(mlmi.predict(), 4)
        err = np.round(np.abs(MI - MI_hat), 2)
        err_list.append((i,err))

# Plot approximation error
plt.plot(*zip(*err_list))plt.title("Mutual Information Estimation Error")
plt.xlabel("Number of samples")
plt.ylabel("Mean estimation error")
plt.show()
```
