import pickle

import numpy as np
from os.path import join, basename, normpath

class TestMerger:
    def __init__(self, path, path_out, mean_std_prefix):
        self.path = join(path, basename(normpath(path)))
        self.path_out = join(path_out, basename(normpath(path_out)))
        self.mean_std_prefix = mean_std_prefix
        self.mean_std_dataset = None
        self.mean_std_good = None

    def merge(self):
        raw_data_train = np.load(f"{self.path}_train.npy").astype(np.float32)
        raw_data_val = np.load(f"{self.path}_val.npy").astype(np.float32)
        raw_data_test = np.load(f"{self.path}_test.npy").astype(np.float32)

        raw_data = np.concatenate((raw_data_train,raw_data_val, raw_data_test), axis=1)

        #with open(f'{self.path}_mean_std.pkl', 'rb') as mean_std_bin_file:
        #    self.mean_std_dataset = pickle.load(mean_std_bin_file)
        #with open(f'./DataTest/{self.mean_std_prefix}_mean_std.pkl', 'rb') as mean_std_bin_file:
        #    self.mean_std_good = pickle.load(mean_std_bin_file)
        #raw_data = (raw_data * self.mean_std_dataset['std'][None, None, :, None] +
        #            self.mean_std_dataset['mean'][None, None, :, None] -
        #            self.mean_std_good['mean'][None, None, :, None]) / self.mean_std_good['std'][None, None, :, None]

        np.save(f"{self.path_out}.npy", np.array(raw_data, dtype=np.float32),
                allow_pickle=False, fix_imports=False)
