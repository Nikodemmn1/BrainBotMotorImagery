import torch
from torch.optim import AdamW
from torch.nn.functional import nll_loss
from torch import nn
from pytorch_lightning import LightningModule
import torchmetrics


class OneDNet(LightningModule):
    def __init__(self, signal_len, classes_count, train_indices=None, val_indices=None, test_indices=None):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(16, 42, kernel_size=(5,), padding='valid'),
            # nn.Dropout2d(p=0.2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),

            nn.AvgPool1d(kernel_size=3),

            nn.Conv1d(42, 84, kernel_size=(5,), padding='valid'),
            # nn.Dropout2d(p=0.2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),

            #nn.AvgPool1d(kernel_size=3),

            # nn.Conv1d(84, 100, kernel_size=(3,), padding='valid'),
            # nn.Dropout2d(p=0.2),
            # nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.avgpool = nn.AdaptiveAvgPool1d(10)

        self.classifier = nn.Sequential(
            nn.Linear(2940, 100),
            # nn.Dropout(p=0.4),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),

            # nn.Linear(100, 10),
            # nn.Dropout(p=0.3),
            # nn.LeakyReLU(negative_slope=0.01, inplace=True),

            nn.Linear(100, classes_count),
            nn.Softmax()
        )

        self.signal_len = signal_len
        self.accuracy = torchmetrics.Accuracy()
        self.indices = (train_indices, val_indices, test_indices)

    def forward(self, x):
        x = torch.tensor(x)
        sc = self.signal_len
        cl = x.size(1)
        if x.size(2) != sc:
            x = torch.stack([_.flatten()[:-abs((x.size(2) * x.size(1)) - (sc * x.size(1)))].reshape(cl, sc) for _ in
                             x.unbind()])  # create squares
        x = self.features(x)
        #print(x.shape)
        x = torch.flatten(x, 1)
        #print(x.shape)
        x = self.classifier(x)
        if x.size(1) == 1:
            x = x.flatten()  # remove channel dim
        return x

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.0001)

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

    def test_step(self, batch, batch_idx):
        data, label = batch
        output = self(data)
        loss = nll_loss(output, label)
        self.log("Test loss", loss)
        self.accuracy(output, label)
        self.log("Test Acc", self.accuracy)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['indices'] = self.indices
