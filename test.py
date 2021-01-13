import numpy as np

x = np.load('deepsigdata.npy')
s = np.concatenate((x[0], x[1]), axis=0)
y = x.tolist()
print(len(y))
print(len(y[0]))
print(s.shape)
c = s[:, 0] + 1j * s[:, 1]
np.save('iq.npy', c)
iq = np.load('iq.npy')
print(iq)