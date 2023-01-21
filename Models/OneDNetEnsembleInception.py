import io
import pstats
import torch
from torch.optim import AdamW
from torch.nn.functional import binary_cross_entropy, dropout2d
from torch import nn
from pytorch_lightning import LightningModule
import torchmetrics
import seaborn as sn
import pandas as pd
import random
from Utilities import augmentations as aug
import numpy as np
import cProfile
import math


class OneDNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_chanels, **kwargs):
        super(OneDNetConvBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_chanels, **kwargs),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class OneDNetInceptionBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_1x1,
            red_3x3,
            out_3x3,
            red_5x5,
            out_5x5,
            out_pool,
    ):
        super(OneDNetInceptionBlock, self).__init__()
        self.branch1 = OneDNetConvBlock(in_channels, out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            OneDNetConvBlock(in_channels, red_3x3, kernel_size=1, padding=0),
            OneDNetConvBlock(red_3x3, out_3x3, kernel_size=(3,3), padding=(1,1)),
        )
        self.branch3 = nn.Sequential(
            OneDNetConvBlock(in_channels, red_5x5, kernel_size=1),
            OneDNetConvBlock(red_5x5, out_5x5, kernel_size=(3,5), padding=(1,2)),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3,3), padding=(1,1), stride=1),
            OneDNetConvBlock(in_channels, out_pool, kernel_size=1),
        )

    def forward(self, x):
        branches = (self.branch1, self.branch2, self.branch3, self.branch4)
        concatenated = torch.cat([branch(x) for branch in branches], 1)
        return concatenated


class OneDNetEnsemble(LightningModule):
    def __init__(self, train_indices=None, val_indices=None, test_indices=None):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 5), padding='same', padding_mode='circular'),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.AvgPool2d(kernel_size=(1, 3)),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=0.7),

            nn.Conv2d(32, 64, kernel_size=(3, 5), padding='valid'),
            # nn.Dropout2d(p=0.2),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.AvgPool2d(kernel_size=(1, 3)),
            nn.BatchNorm2d(64),
            nn.Dropout2d(p=0.6),

            OneDNetInceptionBlock(64, 32, 48, 64, 8, 16, 16),
            nn.AvgPool2d(kernel_size=(1, 3)),
            nn.BatchNorm2d(128),
            nn.Dropout2d(p=0.6),

            OneDNetInceptionBlock(128, 64, 96, 128, 16, 32, 32),
            nn.AvgPool2d(kernel_size=(1, 3)),
            nn.BatchNorm2d(256),
            nn.Dropout2d(p=0.6),

            OneDNetInceptionBlock(256, 128, 192, 256, 32, 64, 64),
            nn.BatchNorm2d(512),
            nn.AvgPool2d(kernel_size=(3, 3)),
            nn.Dropout2d(p=0.6),

            OneDNetInceptionBlock(512, 256, 384, 512, 64, 128, 128),
            nn.BatchNorm2d(1024),
            nn.Dropout2d(p=0.6),
        )

        self.classifier = nn.Sequential(
            nn.Linear(3072, 250),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.BatchNorm1d(250),
            nn.Dropout(p=0.5),

            nn.Linear(250, 120),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.BatchNorm1d(120),
            nn.Dropout(p=0.5),

            nn.Linear(120, 1),
            nn.Sigmoid()
        )

        self.accuracy = torchmetrics.Accuracy("binary")
        self.confusion_matrix_70 = torchmetrics.ConfusionMatrix("binary", threshold=0.7)
        self.confusion_matrix = torchmetrics.ConfusionMatrix("binary")
        self.indices = (train_indices, val_indices, test_indices)

        self.test_metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.Accuracy("binary"),
                torchmetrics.Precision("binary"),
                torchmetrics.Recall("binary"),
                torchmetrics.F1Score("binary"),
            ]
        )

        self.val_metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.Accuracy("binary"),
                torchmetrics.Precision("binary"),
                torchmetrics.Recall("binary"),
                torchmetrics.F1Score("binary"),
            ]
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if x.size(1) == 1:
            x = x.flatten()  # remove channel dim
        return x

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=7e-5, weight_decay=8e-2)

    def training_step(self, batch, batch_idx):
        data, label = batch
        output = self(data)
        loss = binary_cross_entropy(output, label)
        self.log("Training loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.accuracy(output, label)
        self.log("Training accuracy", self.accuracy, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        output = self(data)
        loss = binary_cross_entropy(output, label)
        self.log("Val loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_metrics.update(output, label.type(torch.int))
        return output, label

    def validation_epoch_end(self, validation_data):
        self.log_dict(self.val_metrics.compute(), on_step=False, on_epoch=True)
        self.val_metrics.reset()

        outputs = torch.cat([x[0] for x in validation_data])
        labels = torch.cat([x[1] for x in validation_data])

        conf_matrix = self.confusion_matrix(outputs, labels.type(torch.int))
        conf_matrix70 = self.confusion_matrix70(outputs, labels.type(torch.int))

        df_cm = pd.DataFrame(conf_matrix.cpu(), index=["false", "true"],
                             columns=["false pred", "true pred"])
        df_cm70 = pd.DataFrame(conf_matrix70.cpu(), index=["false", "true"],
                             columns=["false pred", "true pred"])

        sn.set(font_scale=0.7)
        conf_matrix_figure = sn.heatmap(df_cm, annot=True, vmin=0).get_figure()
        conf_matrix_figure70 = sn.heatmap(df_cm70, annot=True, vmin=0).get_figure()
        self.logger.experiment.add_figure('Confusion matrix', conf_matrix_figure, self.current_epoch)
        self.logger.experiment.add_figure('Confusion matrix 70% thresh.', conf_matrix_figure70, self.current_epoch)

    def test_step(self, batch, batch_idx):
        data, label = batch
        output = self(data)
        loss = binary_cross_entropy(output, label)
        self.log("Test loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.test_metrics.update(output, label.type(torch.int))
        return output, label

    def test_epoch_end(self, test_data):
        outputs = torch.cat([x[0] for x in test_data])
        labels = torch.cat([x[1] for x in test_data])

        conf_matrix = self.confusion_matrix(outputs, labels.type(torch.int))
        conf_matrix70 = self.confusion_matrix70(outputs, labels.type(torch.int))

        df_cm = pd.DataFrame(conf_matrix.cpu(), index=["false", "true"],
                             columns=["false pred", "true pred"])
        df_cm70 = pd.DataFrame(conf_matrix70.cpu(), index=["false", "true"],
                             columns=["false pred", "true pred"])

        sn.set(font_scale=0.7)
        conf_matrix_figure = sn.heatmap(df_cm, annot=True, vmin=0).get_figure()
        conf_matrix_figure70 = sn.heatmap(df_cm70, annot=True, vmin=0).get_figure()
        self.logger.experiment.add_figure('Confusion matrix TEST', conf_matrix_figure, self.current_epoch)
        self.logger.experiment.add_figure('Confusion matrix TEST 70% thresh.', conf_matrix_figure70, self.current_epoch)

        test_dict_raw = self.test_metrics.compute()
        test_dict = dict()
        for key, value in test_dict_raw.items():
            test_dict[f"{key} Test"] = value
        self.log_dict(test_dict, on_step=False, on_epoch=True)
        self.test_metrics.reset()

    def on_save_checkpoint(self, checkpoint):
        checkpoint['indices'] = self.indices