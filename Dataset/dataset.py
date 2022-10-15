import math
import numpy as np
from torch.utils.data import Dataset, Subset


class EEGDataset(Dataset):
    def __init__(self, train_file_path, val_file_path, test_file_path, included_classes=None):
        raw_data_train = np.load(train_file_path).astype(np.float32)
        raw_data_val = np.load(val_file_path).astype(np.float32)
        raw_data_test = np.load(test_file_path).astype(np.float32)

        if included_classes is not None:
            raw_data_train = raw_data_train[included_classes, :, :]
            raw_data_val = raw_data_val[included_classes, :, :]
            raw_data_test = raw_data_test[included_classes, :, :]

        self.class_count = raw_data_train.shape[0]

        raw_data_train = raw_data_train.reshape((raw_data_train.shape[1] * raw_data_train.shape[0],
                                                 raw_data_train.shape[2],
                                                 raw_data_train.shape[3]))
        raw_data_val = raw_data_val.reshape((raw_data_val.shape[1] * raw_data_val.shape[0],
                                             raw_data_val.shape[2],
                                             raw_data_val.shape[3]))
        raw_data_test = raw_data_test.reshape((raw_data_test.shape[1] * raw_data_test.shape[0],
                                               raw_data_test.shape[2],
                                               raw_data_test.shape[3]))

        self.train_len = raw_data_train.shape[0]
        self.val_len = raw_data_val.shape[0]
        self.test_len = raw_data_test.shape[0]

        self.data = np.concatenate([raw_data_train, raw_data_val, raw_data_test], axis=0)
        self.data = self.data.reshape((self.data.shape[0], 1, self.data.shape[1], self.data.shape[2]))

        self.labels = np.hstack([
            np.hstack([
                np.full(raw_data_curr.shape[0] // self.class_count, label, dtype=np.int)
                for label in range(self.class_count)
            ])
            for raw_data_curr in [raw_data_train, raw_data_val, raw_data_test]
        ])

    def get_subsets(self):
        train_subset = Subset(self, range(self.train_len))
        val_subset = Subset(self, range(self.train_len, self.train_len + self.val_len))
        test_subset = Subset(self, range(self.train_len + self.val_len, self.train_len + self.val_len + self.test_len))

        return train_subset, val_subset, test_subset

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
