from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, StochasticWeightAveraging, ModelCheckpoint
from Models.OneDNet import OneDNet
from Dataset.dataset import *


def main():
    included_classes = [0, 1, 2]
    included_channels = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # included_channels = range(16)
    # full_dataset = EEGDataset("./DataMerged/DataMerged_train.npy",
    #                           "./DataMerged/DataMerged_val.npy",
    #                           "./DataMerged/DataMerged_test.npy",
    #                           included_classes, included_channels)
    full_dataset = EEGDataset("./DataBDF/Out/Out_train.npy",
                              "./DataBDF/Out/Out_val.npy",
                              "./DataBDF/Out/Out_test.npy",
                              included_classes, included_channels)
    train_dataset, val_dataset, test_dataset = full_dataset.get_subsets()
    train_dataset, val_dataset, test_dataset = full_dataset.get_subsets()
    train_data = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=12)
    val_data = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_data = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = OneDNet(len(included_channels), included_classes, train_dataset.indices,
                    val_dataset.indices, test_dataset.indices)

    #model = OneDNet.load_from_checkpoint(channel_count=len(included_channels),
    #                                     included_classes=included_classes,
    #                                     checkpoint_path="./lightning_logs/version_118/checkpoints/epoch=59-step=17640.ckpt")

    trainer = Trainer(gpus=-1, callbacks=[TQDMProgressBar(refresh_rate=5),
                                          StochasticWeightAveraging(),
                                          ModelCheckpoint(save_weights_only=False,
                                                          monitor="Val loss",
                                                          save_last=True,
                                                          save_top_k=3,
                                                          mode='min')],
                      check_val_every_n_epoch=1, benchmark=True)

    trainer.fit(model, train_data, val_data)

    trainer.test(model, test_data)


if __name__ == "__main__":
    main()
