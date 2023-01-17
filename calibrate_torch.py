from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.callbacks import TQDMProgressBar, StochasticWeightAveraging, ModelCheckpoint
from Models.OneDNet import OneDNet, CalibrationOneDNet
from Dataset.dataset import *

def get_dataloaders(data_paths, included_classes, included_channels):
    train_path, val_path, test_path = data_paths
    full_dataset = EEGDataset(train_path,
                              val_path,
                              test_path,
                              included_classes, included_channels)
    train_dataset, val_dataset, test_dataset = full_dataset.get_subsets()
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    return train_dataloader, val_dataloader, test_dataloader

def merge_dataloaders(source_dataloader, target_dataloader):
    loaders = {'source': source_dataloader, 'target': target_dataloader}
    combined_loader = CombinedLoader(loaders, mode="max_size_cycle")
    return combined_loader

def main():
    included_classes = [0, 1, 2]
    included_channels = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # included_channels = range(16)

    source_domain_paths = ["./DataBDF/OutTraining/Nikodem/Nikodem_train.npy",
                              "./DataBDF/OutTraining/Nikodem/Nikodem_val.npy",
                              "./DataBDF/OutTesting/Nikodem/Nikodem_test.npy"]
    train_dataloader_s, val_dataloader_s, test_dataloader_s = get_dataloaders(source_domain_paths, included_classes, included_channels)

    target_domain_paths = ["./DataBDF/OutTraining/Piotr/Piotr_train.npy",
                           "./DataBDF/OutTraining/Piotr/Piotr_val.npy",
                           "./DataBDF/OutTesting/Piotr/Piotr_test.npy"]
    train_dataloader_t, val_dataloader_t, test_dataloader_t = get_dataloaders(target_domain_paths, included_classes,
                                                                        included_channels)


    model = CalibrationOneDNet(len(included_channels), included_classes, train_dataloader_s.dataset.indices,
                    val_dataloader_s.dataset.indices, test_dataloader_s.dataset.indices)

    model = CalibrationOneDNet.load_from_checkpoint(channel_count=len(included_channels),
                                        included_classes=included_classes,
                                        checkpoint_path="./lightning_logs/version_29/checkpoints/last.ckpt")

    trainer = Trainer(gpus=-1, callbacks=[TQDMProgressBar(refresh_rate=5),
                                          StochasticWeightAveraging(swa_lrs=0.001),
                                          ModelCheckpoint(save_weights_only=False,
                                                          monitor="Val loss",
                                                          save_last=True,
                                                          save_top_k=3,
                                                          mode='min')],
                      check_val_every_n_epoch=1, benchmark=True)
    trainer.fit(model, merge_dataloaders(train_dataloader_s, train_dataloader_t), merge_dataloaders(val_dataloader_s, val_dataloader_t))

    trainer.test(model, merge_dataloaders(test_dataloader_s, test_dataloader_t))


if __name__ == "__main__":
    main()
