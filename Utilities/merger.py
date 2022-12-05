import numpy as np
from os.path import join, basename, normpath

class Merger:
    def __init__(self, path1, path2, path_out):
        self.path1 = join(path1, basename(normpath(path1)))
        self.path2 = join(path2, basename(normpath(path2)))
        self.path_out = join(path_out, basename(normpath(path_out)))

    def merge(self):
        raw_data_train1 = np.load(f"{self.path1}_train.npy").astype(np.float32)
        raw_data_val1 = np.load(f"{self.path1}_train.npy").astype(np.float32)
        raw_data_test1 = np.load(f"{self.path1}_train.npy").astype(np.float32)

        raw_data_train2 = np.load(f"{self.path2}_train.npy").astype(np.float32)
        raw_data_val2 = np.load(f"{self.path2}_train.npy").astype(np.float32)
        raw_data_test2 = np.load(f"{self.path2}_train.npy").astype(np.float32)

        raw_data_train = np.concatenate((raw_data_train1,
                                         raw_data_train2[:3, :, :, :],
                                         raw_data_val2[:3, :, :, :],
                                         raw_data_test2[:3, :, :, :]), axis=1)
        raw_data_val = raw_data_val1
        raw_data_test = raw_data_test1

        np.save(f"{self.path_out}_train.npy", np.array(raw_data_train, dtype=np.float32),
                allow_pickle=False, fix_imports=False)
        np.save(f"{self.path_out}_val.npy", np.array(raw_data_val, dtype=np.float32),
                allow_pickle=False, fix_imports=False)
        np.save(f"{self.path_out}_test.npy", np.array(raw_data_test, dtype=np.float32),
                allow_pickle=False, fix_imports=False)
