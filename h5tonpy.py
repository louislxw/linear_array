import h5py
import numpy as np

read_filename = "deepsig/part5.h5"
write_filename = "data/part5.npy"

with h5py.File(read_filename, "r") as f1:
    # List all groups (X, Y, Z)
    print("Keys: %s" % f1.keys())
    a_group_key = list(f1.keys())[0]

    # Get the data of key[0] = X
    data = list(f1[a_group_key])
    data_clip = np.concatenate((data[0], data[1]), axis=0)
    print(data_clip.shape)
    print(data_clip)
    data_complex = data_clip[:, 0] + 1j * data_clip[:, 1]
    print(data_complex.shape)
    print(data_complex)

with open(write_filename, 'wb') as f2:
    np.save(f2, data_complex)
