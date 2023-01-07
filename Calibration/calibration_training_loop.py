from Dataset.dataset import CalibrationDataset
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, StochasticWeightAveraging, ModelCheckpoint
from Models.OneDNet import OneDNet
from pytorch_lightning.loops import FitLoop
from pytorch_lightning.trainer.supporters import CombinedLoader
import logging
import os
from typing import Any, Optional, Type

import pytorch_lightning as pl
from pytorch_lightning.accelerators import CUDAAccelerator
from pytorch_lightning.loops import Loop
from pytorch_lightning.loops.epoch import TrainingEpochLoop
from pytorch_lightning.loops.epoch.training_epoch_loop import _OUTPUTS_TYPE as _EPOCH_OUTPUTS_TYPE
from pytorch_lightning.loops.utilities import _is_max_limit_reached, _set_sampler_epoch
from pytorch_lightning.trainer.connectors.logger_connector.result import _ResultCollection
from pytorch_lightning.trainer.progress import Progress
from pytorch_lightning.trainer.supporters import CombinedLoader, TensorRunningAccum
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.fetching import (
    AbstractDataFetcher,
    DataFetcher,
    DataLoaderIterDataFetcher,
    InterBatchParallelDataFetcher,
)
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_debug, rank_zero_info, rank_zero_warn
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature

class CalibrationTrainingLoop(FitLoop):
    def __init__(self, included_classes=None, included_channels=None):
        super().__init__()
        self.included_classes = included_classes
        self.included_channels = included_channels

    def _select_data_fetcher(self, trainer: "pl.Trainer") -> Type[AbstractDataFetcher]:
        training_step_fx = getattr(trainer.lightning_module, "training_step")
        if is_param_in_hook_signature(training_step_fx, "dataloader_iter", explicit=True):
            rank_zero_warn(
                "Found `dataloader_iter` argument in the `training_step`. Note that the support for "
                "this signature is experimental and the behavior is subject to change."
            )
            return DataLoaderIterDataFetcher
        elif os.getenv("PL_INTER_BATCH_PARALLELISM", "0") == "1":
            if not isinstance(trainer.accelerator, CUDAAccelerator):
                raise MisconfigurationException("Inter batch parallelism is available only when using Nvidia GPUs.")
            return InterBatchParallelDataFetcher
        return DataFetcher
    def on_advance_end(self) -> None:
        super().on_advance_end()
        self.trainer.train_dataloader.dataset.datasets.dataset.update_dataset()
        # train_dataset, val_dataset, test_dataset = self.trainer.train_dataloader.dataset.datasets.dataset.get_subsets()
        # self.trainer.train_dataloader.dataset.datasets = train_dataset
        # self.trainer.train_dataloader.sampler.data_source.indices = train_dataset.indices
        # self.trainer.val_dataloaders[0].dataset.datasets = val_dataset
        # self.trainer.val_dataloaders[0].sampler.data_source.indices = val_dataset.indices
        self.on_run_start()
        # self.trainer.reset_train_dataloader(self.trainer.lightning_module)
        # # reload the evaluation dataloaders too for proper display in the progress bar
        # if self.epoch_loop._should_check_val_epoch():
        #     self.epoch_loop.val_loop._reload_evaluation_dataloaders()
        #
        # data_fetcher_cls = self._select_data_fetcher(self.trainer)
        # self._data_fetcher = data_fetcher_cls(prefetch_batches=self.prefetch_batches)

def main():
    load_from_checkpoint = False
    included_classes = [0, 1, 2]
    # included_channels = range(16)
    included_channels = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    full_dataset = CalibrationDataset("../calibration_data.npy", "../calibration_labels.npy",
                              included_classes)
    train_dataset, val_dataset, test_dataset = full_dataset.get_subsets()
    train_data = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
    val_data = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0, drop_last=True)
    test_data = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    if load_from_checkpoint:
        model = OneDNet.load_from_checkpoint(channel_count=len(included_channels),
                                             included_classes=included_classes,
                                             checkpoint_path="base_checkpoint.ckpt")
    else:
        model = OneDNet(len(included_channels), included_classes, train_dataset.indices,
                        val_dataset.indices, test_dataset.indices)

    trainer = Trainer(gpus=-1, callbacks=[TQDMProgressBar(refresh_rate=5),
                                          StochasticWeightAveraging(swa_lrs=0.001),
                                          ModelCheckpoint(save_weights_only=False,
                                                          monitor="Val loss",
                                                          save_last=True,
                                                          save_top_k=3,
                                                          mode='min')],
                      check_val_every_n_epoch=1, benchmark=True, log_every_n_steps=40)
    trainer.fit_loop = CalibrationTrainingLoop(included_classes=included_classes, included_channels=included_channels)
    trainer.fit(model, train_data, val_data)

    trainer.test(model, test_data)

if __name__ == "__main__":
    main()