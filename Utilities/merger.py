import pickle

import numpy as np
from os.path import join, basename, normpath, isfile

class Merger:
    def __init__(self, in_paths, out_path):
        self.in_paths = [join(in_path, basename(normpath(in_path))) for in_path in in_paths]
        self.out_path = join(out_path, basename(normpath(out_path)))


    def merge(self):
        for dataset_type in ['train', 'val', 'test']:
            unmerged_data = []
            for in_path in self.in_paths:
                final_in_path = f"{in_path}_{dataset_type}.npy"
                if isfile(final_in_path):
                    unmerged_data += np.load(final_in_path).astype('float32')
            if len(unmerged_data) == 0:
                print(f"Error - no {dataset_type} data provided!")
                return
            else:
                merged_data = np.concatenate(unmerged_data, axis=1)
                out_path = f"{self.out_path}_{dataset_type}.npy"
                np.save(out_path, merged_data, dtype="float32")
