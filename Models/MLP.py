import torch
from torch.optim import AdamW
from torch.nn.functional import nll_loss
from torch import nn
from pytorch_lightning import LightningModule
import torchmetrics
import seaborn as sn
import pandas as pd
class MLP(LightningModule):
    def __init__(self, n_features, classes_count, train_indices=None, val_indices=None, test_indices=None):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            
            nn.Linear(32, 16),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),

            nn.Linear(16, 10),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),

            nn.Linear(10, classes_count),
            nn.Softmax()
        )
        self.accuracy = torchmetrics.Accuracy()
        self.indices = (train_indices, val_indices, test_indices)
        self.confusion_matrix = torchmetrics.ConfusionMatrix(classes_count)
        self.class_names = ["left h",
                            "right h",
                            "passive",
                            "left l",
                            "tongue",
                            "right l"]
        self.class_names = [self.class_names[i] for i in range(classes_count)]
        
    def forward(self, x):
        x = self.classifier(x)
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


    # def validation_epoch_end(self, validation_data):
    #     outputs = torch.cat([torch.max(x[0], 1).indices for x in validation_data])
    #     labels = torch.cat([x[1] for x in validation_data])

    #     conf_matrix = self.confusion_matrix(outputs, labels)

    #     df_cm = pd.DataFrame(conf_matrix.cpu(), index=self.class_names,
    #                          columns=[x + " pred" for x in self.class_names])

    #     sn.set(font_scale=0.7)
    #     conf_matrix_figure = sn.heatmap(df_cm, annot=True).get_figure()
    #     self.logger.experiment.add_figure('Confusion matrix', conf_matrix_figure, self.current_epoch)

    def test_step(self, batch, batch_idx):
        data, label = batch
        output = self(data)
        loss = nll_loss(output, label)
        self.log("Test loss", loss)
        self.accuracy(output, label)
        self.log("Test Acc", self.accuracy)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['indices'] = self.indices
