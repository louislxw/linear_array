import numpy as np
import scipy.io

mydata = np.load('data/part5.npy')
# print(mydata)
scipy.io.savemat('matlab/part5.mat', {'mydata': mydata})
