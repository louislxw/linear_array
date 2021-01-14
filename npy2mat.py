import numpy as np
import scipy.io

mydata = np.load('data/iq.npy')
scipy.io.savemat('data/iq.mat', {'mydata': mydata})
