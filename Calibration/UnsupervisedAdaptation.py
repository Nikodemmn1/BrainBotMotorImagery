from torch.utils.data import DataLoader
from Models.OneDNet import OneDNet
from Dataset.dataset import *
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, StochasticWeightAveraging, ModelCheckpoint
def main():
    included_classes = [0, 1, 2]
    included_channels = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    full_dataset = EEGDataset("../DataBDF/OutNikodem/OutNikodem_train.npy",
                              "../DataBDF/OutNikodem/OutNikodem_val.npy",
                              "../DataBDF/OutNikodem/OutNikodem_test.npy",
                              included_classes, included_channels)
    train_dataset, val_dataset, test_dataset = full_dataset.get_subsets()
    train_data = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=8)
    val_data = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_data = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = OneDNet.load_from_checkpoint(channel_count=len(included_channels),
                                         included_classes=included_classes,
                                         domain_name='target',
                                         checkpoint_path="../lightning_logs/version_20/checkpoints/epoch=39-val_loss=0.00-val_accuracy=0.00.ckpt")

    trainer = Trainer(gpus=-1, callbacks=[TQDMProgressBar(refresh_rate=5),
                                          StochasticWeightAveraging(swa_lrs=1e-2),
                                          ModelCheckpoint(filename="{epoch}-{val_loss:.2f}-{val_accuracy:.2f}",
                                                          save_weights_only=False,
                                                          monitor="Val loss",
                                                          save_last=True,
                                                          save_top_k=3,
                                                          mode='min'),
                                          ModelCheckpoint(filename="{epoch}-{val_accuracy:.2f}-{val_loss:.2f}",
                                                          save_weights_only=False,
                                                          monitor="MulticlassAccuracy",
                                                          save_top_k=3,
                                                          mode='max')],
                      check_val_every_n_epoch=2, benchmark=True, max_epochs=1000000)

    trainer.test(model, train_data)
if __name__ == "__main__":
    main()