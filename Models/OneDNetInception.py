import io
import pstats
import torch
from torch.optim import AdamW
from torch.nn.functional import nll_loss
from torch import nn
from pytorch_lightning import LightningModule
import torchmetrics
import seaborn as sn
import pandas as pd
import random
from Utilities import augmentations as aug
import numpy as np
import cProfile


class OneDNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_chanels, **kwargs):
        super(OneDNetConvBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_chanels, **kwargs),
            nn.BatchNorm2d(out_chanels),
            nn.LeakyReLU(negative_slope=0.05, inplace=True)
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
            OneDNetConvBlock(red_3x3, out_3x3, kernel_size=(1,3), padding=(0,1)),
        )
        self.branch3 = nn.Sequential(
            OneDNetConvBlock(in_channels, red_5x5, kernel_size=1),
            OneDNetConvBlock(red_5x5, out_5x5, kernel_size=(1,5), padding=(0,2)),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1,3), padding=(0,1), stride=1),
            OneDNetConvBlock(in_channels, out_pool, kernel_size=1),
        )

    def forward(self, x):
        branches = (self.branch1, self.branch2, self.branch3, self.branch4)
        return torch.cat([branch(x) for branch in branches], 1)


class OneDNetInception(LightningModule):
    def __init__(self, channel_count, included_classes,
                 train_indices=None, val_indices=None, test_indices=None):
        super().__init__()

        classes_count = len(included_classes)

        self.noise_inject = GaussianNoiseInjector(6, 0.5)

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

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding='valid'),
            # nn.Dropout2d(p=0.2),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.BatchNorm2d(128),
            nn.Dropout2d(p=0.6),

            OneDNetInceptionBlock(128, 64, 32, 64, 16, 32, 32),
            OneDNetInceptionBlock(192, 80, 48, 80, 32, 48, 48),
            #OneDNetInceptionBlock(256, 96, 64, 96, 48, 64, 64),
            #OneDNetInceptionBlock(320, 160, 96, 160, 64, 96, 96),
            nn.AvgPool2d(kernel_size=(1, 2)),
        )

        self.classifier = nn.Sequential(
            nn.Linear(7168, 250),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.BatchNorm1d(250),
            nn.Dropout(p=0.5),

            nn.Linear(250, 120),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.BatchNorm1d(120),
            nn.Dropout(p=0.7),

            nn.Linear(120, classes_count),
            nn.LogSoftmax(dim=1)
        )

        self.accuracy = torchmetrics.Accuracy("multiclass", num_classes=3)
        self.confusion_matrix = torchmetrics.ConfusionMatrix("multiclass", num_classes=3)
        self.indices = (train_indices, val_indices, test_indices)
        self.class_names = ["left h",
                            "right h",
                            "passive",
                            "left l",
                            "tongue",
                            "right l"]
        self.class_names = [self.class_names[i] for i in included_classes]

        self.test_metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.Accuracy("multiclass", num_classes=3),
                torchmetrics.Precision(task='multiclass', num_classes=3, average='macro'),
                torchmetrics.Recall(task='multiclass', num_classes=3, average='macro'),
                torchmetrics.F1Score(task='multiclass', num_classes=3, average='macro'),
            ]
        )

        self.augmenter = Augmenter(perm=True, mag=True, time=True, jitt=True,
                 perm_max_seg=5, mag_std=0.2, mag_knot=4, time_std=0.2, time_knot=4, jitt_std=0.03)

    def forward(self, x):
        # x = self.noise_inject(x)
        x = self.features(x)
        # print(x.size())
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
        loss = nll_loss(output, label)
        self.log("Training loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.accuracy(output, label)
        self.log("Training accuracy", self.accuracy, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        output = self(data)
        output[0] = 0
        output[1] = 0
        loss = nll_loss(output, label)
        self.log("Val loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.accuracy(output, label)
        self.log("Val accuracy", self.accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return output, label

    def validation_epoch_end(self, validation_data):
        outputs = torch.cat([torch.max(x[0], 1).indices for x in validation_data])
        labels = torch.cat([x[1] for x in validation_data])

        conf_matrix = self.confusion_matrix(outputs, labels)

        df_cm = pd.DataFrame(conf_matrix.cpu(), index=self.class_names,
                             columns=[x + " pred" for x in self.class_names])

        sn.set(font_scale=0.7)
        conf_matrix_figure = sn.heatmap(df_cm, annot=True).get_figure()
        self.logger.experiment.add_figure('Confusion matrix', conf_matrix_figure, self.current_epoch)

    def test_step(self, batch, batch_idx):
        data, label = batch
        output = self(data)
        loss = nll_loss(output, label)
        self.log("Test loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.test_metrics.update(output, label)
        return output, label

    def test_epoch_end(self, test_data):
        outputs = torch.cat([torch.max(x[0], 1).indices for x in test_data])
        labels = torch.cat([x[1] for x in test_data])

        conf_matrix = self.confusion_matrix(outputs, labels)

        df_cm = pd.DataFrame(conf_matrix.cpu(), index=self.class_names,
                             columns=[x + " pred" for x in self.class_names])

        sn.set(font_scale=0.7)
        conf_matrix_figure = sn.heatmap(df_cm, annot=True).get_figure()
        self.logger.experiment.add_figure('Confusion matrix TEST', conf_matrix_figure, self.current_epoch)
        self.log_dict(self.test_metrics.compute(), on_step=False, on_epoch=True)
        self.test_metrics.reset()

    def on_save_checkpoint(self, checkpoint):
        checkpoint['indices'] = self.indices


class Augmenter(nn.Module):
    def __init__(self, perm=True, mag=True, time=True, jitt=True,
                 perm_max_seg=5, mag_std=0.2, mag_knot=4, time_std=0.2, time_knot=4, jitt_std=0.03):
        super().__init__()
        self.perm_max_seg = perm_max_seg
        self.mag_std = mag_std
        self.mag_knot = mag_knot
        self.time_std = time_std
        self.time_knot = time_knot
        self.jitt_std = jitt_std
        self.perm = perm
        self.mag = mag
        self.time = time
        self.jitt = jitt

    @torch.no_grad()
    def forward(self, x):
        if self.jitt:
            x = aug.jitter(x, self.jitt_std)
        if self.perm:
            x = aug.permutation(x, self.perm_max_seg)
        if self.mag:
            x = aug.magnitude_warp(x, self.mag_std, self.mag_knot)
        if self.time:
            x = aug.time_warp(x, self.time_std, self.time_knot)
        return x



class GaussianNoiseInjector(nn.Module):
    def __init__(self, snr_db, chance):
        super().__init__()
        self.snr_db = snr_db
        self.chance = chance

    def forward(self, x):
        for b in range(x.shape[0]):
            if random.random() < self.chance:
                for ch in range(x.shape[2]):
                    p_sig = torch.abs(torch.mean(x[b, 0, ch, :]))
                    power = torch.log10(p_sig) - self.snr_db / 10.0
                    noise_std = torch.sqrt(torch.pow(10, power))
                    noise = torch.empty(x.shape[3],
                                        dtype=torch.float32,
                                        device=torch.device('cuda')).normal_(mean=0, std=noise_std)
                    x[b, 0, ch, :] += noise
        return x
