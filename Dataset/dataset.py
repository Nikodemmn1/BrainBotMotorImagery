import math
import numpy as np
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split

TRAIN_PERCENT = 0.7
VAL_PERCENT = 0.15


class EEGDataset(Dataset):
    def __init__(self, data_file_path, memmap=False):
        if memmap:
            raw_data = np.memmap(data_file_path, dtype=np.float32, mode='c')
        else:
            raw_data = np.load(data_file_path).astype(np.float32)
        self.labels = np.hstack([np.full(raw_data.shape[1], label, dtype=np.int) for label in range(raw_data.shape[0])])
        self.data = raw_data.reshape((raw_data.shape[0] * raw_data.shape[1], raw_data.shape[2], raw_data.shape[3]))
        self.class_count = raw_data.shape[0]

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def dataset_split_train_val_test(dataset):
    train_indices, val_test_indices, _, val_test_labels = train_test_split(list(range(len(dataset.labels))),
                                                                           dataset.labels,
                                                                           test_size=1 - TRAIN_PERCENT,
                                                                           stratify=dataset.labels)
    val_indices, test_indices, *rest = train_test_split(val_test_indices, val_test_labels,
                                                        train_size=VAL_PERCENT / (1 - TRAIN_PERCENT),
                                                        stratify=val_test_labels)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset


def dataset_split_train_test(dataset):
    train_indices, test_indices = train_test_split(list(range(len(dataset.labels))),
                                                   dataset.labels,
                                                   test_size=1 - TRAIN_PERCENT,
                                                   stratify=dataset.labels)

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, test_dataset
