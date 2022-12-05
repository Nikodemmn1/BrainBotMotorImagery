import torch
from torch.optim import AdamW
from torch.nn.functional import nll_loss
from torch import nn
from pytorch_lightning import LightningModule
import torchmetrics
import seaborn as sn
import pandas as pd
import random


class OneDNet(LightningModule):
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
        )

        self.classifier = nn.Sequential(
            nn.Linear(17920, 250),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.BatchNorm1d(250),
            nn.Dropout(p=0.5),

            nn.Linear(250, 120),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.BatchNorm1d(120),
            nn.Dropout(p=0.5),

            nn.Linear(120, classes_count),
            nn.LogSoftmax(dim=1)
        )

        self.accuracy = torchmetrics.Accuracy()
        self.confusion_matrix = torchmetrics.ConfusionMatrix(classes_count)
        self.indices = (train_indices, val_indices, test_indices)
        self.class_names = ["left h",
                            "right h",
                            "passive",
                            "left l",
                            "tongue",
                            "right l"]
        self.class_names = [self.class_names[i] for i in included_classes]

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
        accuracy = self.accuracy(output, label)
        self.log("Training accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        output = self(data)
        output[0] = 0
        output[1] = 0
        loss = nll_loss(output, label)
        self.log("Val loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        accuracy = self.accuracy(output, label)
        self.log("Val accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)
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
        accuracy = self.accuracy(output, label)
        self.log("Test Acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)
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

    def on_save_checkpoint(self, checkpoint):
        checkpoint['indices'] = self.indices


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
