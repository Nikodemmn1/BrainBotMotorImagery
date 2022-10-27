import math
import numpy as np
from torch.utils.data import Dataset, Subset
import random
import copy
from tqdm import tqdm

class EEGDataset(Dataset):
    def __init__(self, train_file_path, val_file_path, test_file_path, included_classes=None, included_channels=None, augment=False, augmentations_num = 9):
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
        self.channels_count = raw_data_train.shape[2]

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

        self.labels = np.hstack([
            np.hstack([
                np.full(raw_data_curr.shape[0] // self.class_count, label, dtype=np.int)
                for label in range(self.class_count)
            ])
            for raw_data_curr in [raw_data_train, raw_data_val, raw_data_test]
        ])
        
        if augment == True:
            self.augmentations_num = augmentations_num
            self.augmented_data = copy.deepcopy(raw_data_train)
            self.augmented_labels = copy.deepcopy(self.labels[range(self.train_len)])
            val_labels = self.labels[range(self.train_len, self.train_len + self.val_len)]
            test_labels = self.labels[range(self.train_len + self.val_len, self.train_len + self.val_len + self.test_len)]
            self.augmented_len = self.train_len
            self.augment_data()
            self.data = np.concatenate([self.augmented_data, raw_data_val, raw_data_test])
            self.train_len = self.augmented_len
            self.labels = np.concatenate([self.augmented_labels, val_labels, test_labels])

    def get_subsets(self):
        train_subset = Subset(self, range(self.train_len))
        val_subset = Subset(self, range(self.train_len, self.train_len + self.val_len))
        test_subset = Subset(self, range(self.train_len + self.val_len, self.train_len + self.val_len + self.test_len))

        return train_subset, val_subset, test_subset
    
    def augment_data(self):
        print("Augmenting data...")
        for i in tqdm(range(self.augmentations_num)):
            augmented_data = copy.deepcopy(self.data[range(self.train_len)])
            augmented_data = self.amplify_signal(augmented_data)
            augmented_data = self.inverse_polarity(augmented_data)
            augmented_data = self.rotate_along_time(augmented_data)
            augmented_data = self.add_noise(augmented_data)
            self.insert_augmented_data(augmented_data)        

    def amplify_signal(self, augmented_data):
        gain = random.uniform(0.2, 5.0)
        print("Signal amplification... Gain = ", gain)
        augmented_data = self.data[range(self.train_len)]*gain
        return augmented_data
    
    def inverse_polarity(self, augmented_data):
        #polarity is changed channel-wise
        print("Changing polarity channel-wise")
        for i in range(self.channels_count):
            polarity = np.random.choice([-1, 1])
            augmented_data[:, i, :] = augmented_data[:, i, :]*polarity
        return augmented_data

    def rotate_along_time(self, augmented_data):
        print("Rotating along time...")
        roll_size = random.randint(-self.data.shape[2] // 2,  self.data.shape[2] // 2)
        augmented_data = np.roll(augmented_data, roll_size, axis = 2)
        return augmented_data

    def add_noise(self, augmented_data):
        print("Adding noise...")
        noise = np.random.normal(loc=0.0, scale = 0.01, size=(1, self.channels_count, 1))
        augmented_data = augmented_data + noise
        return augmented_data
        
    def insert_augmented_data(self, new_data):
        self.augmented_data = np.insert(self.augmented_data, self.augmented_len, new_data, axis = 0)
        new_data_labels = self.labels[range(self.train_len)]
        self.augmented_labels = np.insert(self.augmented_labels, self.augmented_len, new_data_labels, axis = 0)
        self.augmented_len = self.augmented_data.shape[0]
        return self.augmented_data,  self.augmented_labels, self.augmented_len

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
        

class processedDataset(Dataset):
    def __init__(self, train_file_path, val_file_path, test_file_path):
        data_train = np.load(train_file_path).astype(np.float32)
        data_val = np.load(val_file_path).astype(np.float32)
        data_test = np.load(test_file_path).astype(np.float32)

        labels_train = data_train[:, -1].astype(np.int)
        labels_val = data_val[:, -1].astype(np.int)
        labels_test = data_test[:, -1].astype(np.int)
        raw_data_train = data_train[:, :-1]
        raw_data_val = data_val[:, :-1]
        raw_data_test = data_test[:, :-1]

        self.train_len = raw_data_train.shape[0]
        self.val_len = raw_data_val.shape[0]
        self.test_len = raw_data_test.shape[0]

        self.data = np.concatenate([raw_data_train, raw_data_val, raw_data_test], axis=0)

        self.labels = np.concatenate([labels_train, labels_val, labels_test], axis=0)

    def get_subsets(self):
        train_subset = Subset(self, range(self.train_len))
        val_subset = Subset(self, range(self.train_len, self.train_len + self.val_len))
        test_subset = Subset(self, range(self.train_len + self.val_len, self.train_len + self.val_len + self.test_len))

        return train_subset, val_subset, test_subset

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
