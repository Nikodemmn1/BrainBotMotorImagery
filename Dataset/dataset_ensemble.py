import math
import numpy as np
from torch.utils.data import Dataset, Subset
from Utilities.calibration_funcs import load_calibration_data

class EEGDatasetEnsemble(Dataset):
    def __init__(self, train_file_path, val_file_path, test_file_path, positive_class=None, included_channels=None):
        if type(train_file_path) is list:
            raw_datas = []
            for p in train_file_path:
                raw_datas.append(np.load(p).astype(np.float32))
            shortest_len = min([raw_data.shape[3] for raw_data in raw_datas])
            raw_datas = [raw_data[:, :, :, :shortest_len] for raw_data in raw_datas]
            raw_data_train = np.concatenate(raw_datas, axis=1)
            raw_data_val = np.load(val_file_path).astype(np.float32)[:, :, :, :shortest_len]
            raw_data_test = np.load(test_file_path).astype(np.float32)[:, :, :, :shortest_len]
        else:
            raw_data_train = np.load(train_file_path).astype(np.float32)
            raw_data_val = np.load(val_file_path).astype(np.float32)
            raw_data_test = np.load(test_file_path).astype(np.float32)

        if included_channels is not None:
            raw_data_train = raw_data_train[:, :, included_channels, :]
            raw_data_val = raw_data_val[:, :, included_channels, :]
            raw_data_test = raw_data_test[:, :, included_channels, :]

        self.class_count = raw_data_train.shape[0]

        raw_data_train[[0, positive_class], ...] = raw_data_train[[positive_class, 0], ...]
        raw_data_test[[0, positive_class], ...] = raw_data_test[[positive_class, 0], ...]
        raw_data_val[[0, positive_class], ...] = raw_data_val[[positive_class, 0], ...]

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
                np.full(raw_data_curr.shape[0] // self.class_count, 1, dtype=np.int),
                np.full((raw_data_curr.shape[0] // self.class_count) * (self.class_count - 1), 0, dtype=np.int)
            ])
            for raw_data_curr in [raw_data_train, raw_data_val, raw_data_test]
        ]).astype('float32')

        neg_weight = 1 / (self.class_count * 2 * self.labels.shape[0] / (self.class_count - 1))
        pos_weight = neg_weight * (self.class_count - 1)
        self.weights = np.where(self.labels == 1,
                                np.full_like(self.labels, pos_weight, dtype='float32'),
                                np.full_like(self.labels, neg_weight, dtype='float32'))
        self.weights_train = self.weights[range(self.train_len)]

    def get_subsets(self):
        train_subset = Subset(self, range(self.train_len))
        val_subset = Subset(self, [*range(self.train_len, self.train_len + self.val_len // self.class_count)] +
                            [*range(self.train_len + self.val_len // self.class_count, self.train_len + self.val_len, self.class_count-1)])
        test_subset = Subset(self, [*range(self.train_len + self.val_len, self.train_len + self.val_len + self.test_len // self.class_count)] +
                            [*range(self.train_len + self.val_len + self.test_len // self.class_count, self.train_len + self.val_len + self.test_len, self.class_count-1)])

        return train_subset, val_subset, test_subset

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
