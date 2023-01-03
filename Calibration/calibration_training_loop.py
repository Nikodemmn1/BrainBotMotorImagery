from Dataset.dataset import CalibrationDataset
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, StochasticWeightAveraging, ModelCheckpoint
from Models.OneDNet import OneDNet
from pytorch_lightning.loops import FitLoop
from pytorch_lightning.trainer.supporters import CombinedLoader

class CalibrationTrainingLoop(FitLoop):
    def __init__(self, included_classes=None, included_channels=None):
        super().__init__()
        self.included_classes = included_classes
        self.included_channels = included_channels

    def on_advance_end(self) -> None:
        ## trainer still uses old data
        self.trainer.train_dataloader.dataset.datasets.dataset.update_dataset()
        train_dataset, val_dataset, test_dataset = self.trainer.train_dataloader.dataset.datasets.dataset.get_subsets()
        self.trainer.train_dataloader.dataset.datasets = train_dataset
        self.trainer.train_dataloader.sampler.data_source.indices = train_dataset.indices
        super().on_advance_end()
        self.on_run_start()

def main():
    included_classes = [0, 1, 2]
    # included_channels = range(16)
    included_channels = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    full_dataset = CalibrationDataset("../calibration_data.npy", "../calibration_labels.npy",
                              included_classes)
    train_dataset, val_dataset, test_dataset = full_dataset.get_subsets()
    train_data = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_data = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_data = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    model = OneDNet(len(included_channels), included_classes, train_dataset.indices,
                    val_dataset.indices, test_dataset.indices)

    trainer = Trainer(gpus=-1, callbacks=[TQDMProgressBar(refresh_rate=5),
                                          StochasticWeightAveraging(swa_lrs=0.001),
                                          ModelCheckpoint(save_weights_only=False,
                                                          monitor="Val loss",
                                                          save_last=True,
                                                          save_top_k=3,
                                                          mode='min')],
                      check_val_every_n_epoch=1, benchmark=True)
    trainer.fit_loop = CalibrationTrainingLoop(included_classes=included_classes, included_channels=included_channels)
    trainer.fit(model, train_data, val_data)

    trainer.test(model, test_data)

if __name__ == "__main__":
    main()