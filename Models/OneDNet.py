import torch
from torch.optim import AdamW
from torch.nn.functional import nll_loss
from torch import nn
from pytorch_lightning import LightningModule
import torchmetrics
import seaborn as sn
import pandas as pd


class OneDNet(LightningModule):
    def __init__(self, signal_len, classes_count, included_classes,
                 train_indices=None, val_indices=None, test_indices=None):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding='same', padding_mode='circular'),
            # nn.Dropout2d(p=0.2),
            # nn.LeakyReLU(negative_slope=0.01, inplace=True),

            nn.Conv2d(16, 16, kernel_size=(1, 3), padding='valid'),
            # nn.Dropout2d(p=0.2),
            # nn.LeakyReLU(negative_slope=0.01, inplace=True),

            nn.AvgPool2d(kernel_size=(1, 3)),

            nn.Conv2d(16, 32, kernel_size=(1, 3), padding='valid'),
            # nn.Dropout2d(p=0.2),
            # nn.LeakyReLU(negative_slope=0.01, inplace=True),

            nn.Conv2d(32, 32, kernel_size=(1, 3), padding='valid'),
            # nn.Dropout2d(p=0.2),
            # nn.LeakyReLU(negative_slope=0.01, inplace=True),

            nn.Conv2d(32, 32, kernel_size=(1, 3), padding='valid'),
            # nn.Dropout2d(p=0.2),
            # nn.LeakyReLU(negative_slope=0.01, inplace=True),

            nn.AvgPool2d(kernel_size=(1, 3)),

            nn.Conv2d(32, 64, kernel_size=(1, 3), padding='valid'),
            # nn.Dropout2d(p=0.2),
            # nn.LeakyReLU(negative_slope=0.01, inplace=True),

            nn.Conv2d(64, 64, kernel_size=(1, 3), padding='valid'),
            # nn.Dropout2d(p=0.2),
            # nn.LeakyReLU(negative_slope=0.01, inplace=True),

            nn.Conv2d(64, 128, kernel_size=(1, 3), padding='valid'),
            # nn.Dropout2d(p=0.2),
            # nn.LeakyReLU(negative_slope=0.01, inplace=True),

            nn.Conv2d(128, 256, kernel_size=(3, 3), padding='valid'),
            # nn.Dropout2d(p=0.2),
            # nn.LeakyReLU(negative_slope=0.01, inplace=True),

            nn.AvgPool2d(kernel_size=(1, 3)),

            # nn.Conv2d(160, 230, kernel_size=(3, 3), padding='same', padding_mode='circular'),
            # nn.Dropout2d(p=0.2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.avgpool = nn.AdaptiveAvgPool1d(10)

        self.classifier = nn.Sequential(
            nn.Linear(93184, classes_count),
            # nn.Dropout(p=0.4),
            # nn.LeakyReLU(negative_slope=0.01, inplace=True),

            #nn.Linear(100, classes_count),
            nn.Softmax()
        )

        self.signal_len = signal_len
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
        x = self.features(x)
        # print(x.size())
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if x.size(1) == 1:
            x = x.flatten()  # remove channel dim
        return x

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.000001, weight_decay=1)

    def training_step(self, batch, batch_idx):
        data, label = batch
        output = self(data)
        loss = nll_loss(output, label)
        return loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        output = self(data)
        loss = nll_loss(output, label)
        self.log("Val loss", loss)
        accuracy = self.accuracy(output, label)
        self.log("Val accuracy", accuracy)
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
        self.log("Test loss", loss)
        self.accuracy(output, label)
        self.log("Test Acc", self.accuracy)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['indices'] = self.indices
