from scipy.signal import welch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, StochasticWeightAveraging, ModelCheckpoint
from Models.OneDNet import OneDNet
from Dataset.dataset import *

included_classes = [0, 1, 2]
included_channels = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
full_dataset = EEGDataset("./DataBDF/Out/Out_train.npy",
                          "./DataBDF/Out/Out_val.npy",
                          "./DataBDF/Out/Out_test.npy",
                          included_classes, included_channels)
train_dataset, val_dataset, test_dataset = full_dataset.get_subsets()
train_data = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=12)
val_data = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
test_data = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

train_data_raw = [(d[0][0, ...], d[1]) for d in train_dataset]
train_data_welch = [(welch(train_d[0], 2048/30, nperseg=107, scaling='spectrum'), train_d[1]) for train_d in train_data_raw]
alpha_indices = np.intersect1d(np.where(7 < train_data_welch[0][0][0]), np.where(train_data_welch[0][0][0] < 12))
beta_indices = np.intersect1d(np.where(12 < train_data_welch[0][0][0]), np.where(train_data_welch[0][0][0] < 35))
data_a_b = [{'a': np.sum(d[0][1][:, alpha_indices], axis=1),
             'b': np.sum(d[0][1][:, beta_indices], axis=1),
             'l': d[1]} for d in train_data_welch]
data_a_b = [((d['a'] < d['b']).astype('int'), d['l']) for d in data_a_b]

data_a_b_left = [np.sum(d[0]) for d in data_a_b if d[1] == 0]
data_a_b_right = [np.sum(d[0]) for d in data_a_b if d[1] == 1]
data_a_b_passive = [np.sum(d[0]) for d in data_a_b if d[1] == 2]

print(data_a_b)