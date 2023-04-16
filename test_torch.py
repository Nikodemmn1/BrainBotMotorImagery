import pickle

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from Models.OneDNet import OneDNet
from Models.OneDNetInception import OneDNetInception
from Dataset.dataset import *
import torch

RENORMALIZATION = False
TEST_DATA_PATH = "DataBDF/OutNikodem/Out_test.npy"
TRAIN_MEAN_STD_PATH = "DataBDF/OutNikodem/Out_mean_std.pkl"
TEST_MEAN_STD_PATH = "DataBDF/OutNikodem/Out_mean_std.pkl"

def renormalize_data(dataset):
    with open(TEST_MEAN_STD_PATH, "rb") as mean_std_file_test:
        mean_std_test = pickle.load(mean_std_file_test)
    mean_test = mean_std_test["mean"]
    std_test = mean_std_test["std"]
    with open(TRAIN_MEAN_STD_PATH, "rb") as mean_std_file_train:
        mean_std_train = pickle.load(mean_std_file_train)
    mean_train = mean_std_train["mean"]
    std_train = mean_std_train["std"]

    dataset.data = (dataset.data * std_test[None, None, :, None] + mean_test[None, None, :, None]) - \
                   mean_train[None, None, :, None] / std_train[None, None, :, None]

def main():
    included_classes = [0, 1, 2]
    included_channels = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    full_dataset = EEGDataset(TEST_DATA_PATH,
                              TEST_DATA_PATH,
                              TEST_DATA_PATH,
                              included_classes, included_channels)
    if RENORMALIZATION:
        renormalize_data(full_dataset)
    _, _, test_dataset = full_dataset.get_subsets()
    test_data = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    model = OneDNet.load_from_checkpoint(channel_count=len(included_channels),
                                         included_classes=included_classes,
                                         checkpoint_path="./model_for_test.ckpt")
    # model = torch.load("./model_nikodem.pt")
    trainer = Trainer(gpus=-1)
    trainer.test(model, test_data)


if __name__ == "__main__":
    main()
