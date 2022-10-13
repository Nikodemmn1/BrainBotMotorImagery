from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, StochasticWeightAveraging
from Models.OneDNet import OneDNet
from Dataset.dataset import *


def main():
    full_dataset = EEGDataset("./Data/EEGLarge/Train/Train.npy",
                              "./Data/EEGLarge/Val/Val.npy",
                              "./Data/EEGLarge/Test/Test.npy")
    train_dataset, val_dataset, test_dataset = full_dataset.get_subsets()
    train_data = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=12)
    val_data = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=12)

    model = OneDNet(full_dataset[0][0].shape[1], full_dataset.class_count, train_dataset.indices, val_dataset.indices,
                    test_dataset.indices)
    trainer = Trainer(gpus=-1, callbacks=[TQDMProgressBar(refresh_rate=100), StochasticWeightAveraging(swa_lrs=1e-2)],
                      check_val_every_n_epoch=5, benchmark=True)

    trainer.fit(model, train_data, val_data)

    trainer.test(model, val_data)


if __name__ == "__main__":
    main()
