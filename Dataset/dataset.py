import math
import numpy as np
from torch.utils.data import Dataset, Subset
from Utilities.calibration_funcs import load_calibration_data

class EEGDataset(Dataset):
    def __init__(self, train_file_path, val_file_path, test_file_path, included_classes=None, included_channels=None):
        raw_data_train = np.load(train_file_path).astype(np.float32)
        raw_data_val = np.load(val_file_path).astype(np.float32)
        raw_data_test = np.load(test_file_path).astype(np.float32)

        if included_classes is not None:
            raw_data_train = raw_data_train[included_classes, :, :]
            raw_data_val = raw_data_val[included_classes, :, :]
            raw_data_test = raw_data_test[included_classes, :, :]

        if included_channels is not None:
            raw_data_train = raw_data_train[:, :, included_channels, :]
            raw_data_val = raw_data_val[:, :, included_channels, :]
            raw_data_test = raw_data_test[:, :, included_channels, :]

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

class CalibrationDataset(EEGDataset):
    def __init__(self, calibration_data_path, calibration_labels_path, included_classes=None, included_channels=None,
                 train_split = 0.7, val_split = 0.2, test_split = 0.1):
        self.data_path = calibration_data_path
        self.labels_path = calibration_labels_path
        self.included_classes = included_classes
        self.included_channels = included_channels
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

        raw_data, labels = load_calibration_data(calibration_data_path, calibration_labels_path)
        self.prepare_data(raw_data, labels)

        self.train_range = range(self.train_len)
        self.val_range = range(self.train_len, self.train_len + self.val_len)
        self.test_range = range(self.train_len + self.val_len, self.train_len + self.val_len + self.test_len)

        self.train_subset = Subset(self, self.train_range)
        self.val_subset = Subset(self, self.val_range)
        self.test_subset = Subset(self, self.test_range)

    def prepare_data(self, raw_data, labels):
        if self.included_classes is not None:
            conditions_list = [labels == class_id for class_id in self.included_classes]
            included_classes_indices = np.where(np.logical_or.reduce(conditions_list))[0]
            raw_data = raw_data[included_classes_indices, :, :, :]
            labels = labels[included_classes_indices]

        if self.included_channels is not None:
            raw_data = raw_data[:, :, self.included_channels, :]

        labels = labels.astype(int)
        self.class_count = np.unique(labels).shape[0]

        indices_pool = list(range(labels.shape[0]))
        np.random.shuffle(indices_pool)
        train_indices = indices_pool[:int(len(indices_pool)*self.train_split)]
        val_indices = indices_pool[int(len(indices_pool)*self.train_split):int(len(indices_pool)*(self.train_split + self.val_split))]
        test_indices = indices_pool[int(len(indices_pool)*(self.train_split + self.val_split)):]
        train_data = raw_data[train_indices]
        val_data = raw_data[val_indices]
        test_data = raw_data[test_indices]

        self.data = np.concatenate([train_data, val_data, test_data], axis=0)
        self.labels = labels

        self.train_len = train_data.shape[0]
        self.val_len = val_data.shape[0]
        self.test_len = test_data.shape[0]

    def update_dataset(self):
        data = load_calibration_data(self.data_path, self.labels_path)
        if data == None:
            return
        else:
            raw_data, labels = data
        self.prepare_data(raw_data, labels)
        self.train_range = range(self.train_len)
        self.val_range = range(self.train_len, self.train_len + self.val_len)
        self.test_range = range(self.train_len + self.val_len, self.train_len + self.val_len + self.test_len)
        self.train_subset.indices = self.train_range
        self.val_subset.indices = self.val_range
        self.test_subset.indices = self.test_range

    def get_subsets(self):
        return self.train_subset, self.val_subset, self.test_subset
