import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, StochasticWeightAveraging
from Models.OneDNet import OneDNet
from Dataset.dataset import *


def main():
    processed_dataset = processedDataset("./processed_data_train.npy",
                                  "./processed_data_val.npy", 
                                  "./processed_data_test.npy")
    train_dataset, val_dataset, test_dataset = processed_dataset.get_subsets()
    trainer = Trainer(callbacks=[TQDMProgressBar(refresh_rate=100), StochasticWeightAveraging(swa_lrs=1e-2)],
                      check_val_every_n_epoch=20, benchmark=True)

    test_indices = torch.load("./checkpoint.ckpt")['indices'][2]
    model = OneDNet.load_from_checkpoint("./checkpoint.ckpt", signal_len=test_dataset[0][0].shape[1],
                                         classes_count=test_dataset.class_count)

    test_data = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=12)
    trainer.test(model, test_data, verbose=True)


if __name__ == "__main__":
    main()
