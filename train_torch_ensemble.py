from torch.utils.data import DataLoader, WeightedRandomSampler
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, StochasticWeightAveraging, ModelCheckpoint
from Models.OneDNet import OneDNet
from Models.OneDNetEnsemble import OneDNetEnsemble
from Models.OneDNetInception import OneDNetInception
from Dataset.dataset_ensemble import *


def main():
    included_classes = [0, 1, 2]
    included_channels = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    full_dataset = EEGDatasetEnsemble("./DataBDF/Out/Out_train.npy",
                              "./DataBDF/Out/Out_val.npy",
                              "./DataBDF/Out/Out_test.npy",
                              1, included_channels)
    train_dataset, val_dataset, test_dataset = full_dataset.get_subsets()
    sampler_train = WeightedRandomSampler(full_dataset.weights_train, len(train_dataset))
    train_data = DataLoader(train_dataset, batch_size=512, num_workers=12, sampler=sampler_train)
    val_data = DataLoader(val_dataset, batch_size=64, num_workers=0, shuffle=False)
    test_data = DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False)

    model = OneDNetEnsemble(train_dataset.indices, val_dataset.indices, test_dataset.indices)

    #model = OneDNet.load_from_checkpoint(channel_count=len(included_channels),
    #                                     included_classes=included_classes,
    #                                     checkpoint_path="./lightning_logs/version_76/checkpoints/epoch=79-step=22960.ckpt")

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
                                                          monitor="BinaryAccuracy",
                                                          save_top_k=3,
                                                          mode='max')],
                      check_val_every_n_epoch=2, benchmark=True, max_epochs=1000000)

    trainer.fit(model, train_data, val_data)

    trainer.test(model, test_data)


if __name__ == "__main__":
    main()
