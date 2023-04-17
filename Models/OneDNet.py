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
import math

class OneDNet(LightningModule):
    def __init__(self, included_classes, domain_name='source', train_indices=None, val_indices=None, test_indices=None):
        super().__init__()
        classes_count = len(included_classes)

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 5), padding='same', padding_mode='circular'),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.AvgPool2d(kernel_size=(1, 3)),
            AdaBatchNorm2d(32, domain_name=domain_name),
            nn.Dropout2d(p=0.7),

            nn.Conv2d(32, 64, kernel_size=(3, 5), padding='valid'),
            # nn.Dropout2d(p=0.2),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.AvgPool2d(kernel_size=(1, 3)),
            AdaBatchNorm2d(64, domain_name=domain_name),
            nn.Dropout2d(p=0.6),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding='valid'),
            # nn.Dropout2d(p=0.2),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            AdaBatchNorm2d(128, domain_name=domain_name),
            nn.Dropout2d(p=0.6),
        )

        self.classifier = nn.Sequential(
            nn.Linear(7168, 250),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            AdaBatchNorm1d(250, domain_name=domain_name),
            nn.Dropout(p=0.5),

            nn.Linear(250, 120),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            AdaBatchNorm1d(120, domain_name=domain_name),
            nn.Dropout(p=0.5),

            nn.Linear(120, classes_count),
            nn.LogSoftmax(dim=1)
        )

        self.accuracy = torchmetrics.Accuracy("multiclass", num_classes=3)
        self.confusion_matrix70 = torchmetrics.ConfusionMatrix("multiclass", num_classes=3, threshold=0.7)
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

        self.val_metrics = torchmetrics.MetricCollection(
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
        loss = nll_loss(output, label)
        self.log("Training loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.accuracy(output, label)
        self.log("Training accuracy", self.accuracy, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        output = self(data)
        loss = nll_loss(output, label)
        self.log("Val loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_metrics.update(output, label.type(torch.int))
        return output, label

    def validation_epoch_end(self, validation_data):
        self.log_dict(self.val_metrics.compute(), on_step=False, on_epoch=True)
        self.val_metrics.reset()

        outputs = torch.cat([torch.max(x[0], 1).indices for x in validation_data])
        labels = torch.cat([x[1] for x in validation_data])

        conf_matrix = self.confusion_matrix(outputs, labels)
        conf_matrix70 = self.confusion_matrix70(outputs, labels)

        df_cm = pd.DataFrame(conf_matrix.cpu(), index=self.class_names,
                             columns=[x + " pred" for x in self.class_names])
        df_cm70 = pd.DataFrame(conf_matrix70.cpu(), index=self.class_names,
                             columns=[x + " pred" for x in self.class_names])

        sn.set(font_scale=0.7)
        conf_matrix_figure = sn.heatmap(df_cm, annot=True).get_figure()
        self.logger.experiment.add_figure('Confusion matrix', conf_matrix_figure, self.current_epoch)
        conf_matrix_figure70 = sn.heatmap(df_cm70, annot=True).get_figure()
        self.logger.experiment.add_figure('Confusion matrix 70% thresh.', conf_matrix_figure70, self.current_epoch)

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
        conf_matrix70 = self.confusion_matrix70(outputs, labels)

        df_cm = pd.DataFrame(conf_matrix.cpu(), index=self.class_names,
                             columns=[x + " pred" for x in self.class_names])
        df_cm70 = pd.DataFrame(conf_matrix70.cpu(), index=self.class_names,
                             columns=[x + " pred" for x in self.class_names])

        sn.set(font_scale=0.7)
        conf_matrix_figure = sn.heatmap(df_cm, annot=True).get_figure()
        self.logger.experiment.add_figure('Confusion matrix TEST', conf_matrix_figure, self.current_epoch)
        conf_matrix_figure70 = sn.heatmap(df_cm70, annot=True).get_figure()
        self.logger.experiment.add_figure('Confusion matrix TEST 70% thresh.', conf_matrix_figure70, self.current_epoch)

        test_dict_raw = self.test_metrics.compute()
        test_dict = dict()
        for key, value in test_dict_raw.items():
            test_dict[f"{key} Test"] = value
        self.log_dict(test_dict, on_step=False, on_epoch=True)
        self.test_metrics.reset()

    def on_save_checkpoint(self, checkpoint):
        checkpoint['indices'] = self.indices

    def on_before_batch_transfer(self, batch, dataloader_idx):
        x, y = batch
        if self.trainer.training and random.random() < -10:
            x = np.swapaxes(np.squeeze(x), 1, 2)

            #prof = cProfile.Profile()
            #prof.enable()
            ##
            x = self.augmenter(x)
            ##
            #prof.disable()
            #s = io.StringIO()
            #sortby = pstats.SortKey.CUMULATIVE
            #ps = pstats.Stats(prof, stream=s).sort_stats(sortby)
            #ps.print_stats()
            #print(s.getvalue())

            x = np.expand_dims(np.swapaxes(x, 1, 2), 1)
            x = torch.Tensor(x).type(torch.float32)
        return x, y


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

class AdaBatchNorm2d(nn.Module):
    def __init__(self, num_features, domain_name, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, eps, momentum, affine)
        self.num_features = num_features
        self.current_domain_name = domain_name
        self.register_buffer('source_running_mean', torch.zeros(self.num_features),
                             persistent=True)
        self.register_buffer('source_running_var', torch.ones(self.num_features), persistent=True)
        self.register_buffer('source_num_samples_tracked', torch.zeros(1), persistent=True)

        self.register_buffer('target_running_mean', torch.zeros(self.num_features),
                             persistent=True)
        self.register_buffer('target_running_var', torch.ones(self.num_features), persistent=True)
        self.register_buffer('target_num_samples_tracked', torch.zeros(1), persistent=True)

        self.registered_domains = []
    def forward(self, x):
        batch_size = x.shape[0]
        device = x.device
        if self.current_domain_name not in self.registered_domains:
            self.registered_domains.append(self.current_domain_name)
        if self.training:
            y = self.bn(x)
            if self.current_domain_name == 'source':
                setattr(self, self.current_domain_name + '_running_mean', self.bn.running_mean)
                setattr(self, self.current_domain_name + '_running_var', self.bn.running_var)
                setattr(self, self.current_domain_name + '_num_samples_tracked', torch.tensor([self.bn.num_batches_tracked*batch_size]).to(device))
        else:
            if self.current_domain_name != 'source':
                self.update_domain_stats(x)
            self.bn.running_mean = getattr(self, self.current_domain_name + '_running_mean').to(device)
            self.bn.running_var = getattr(self, self.current_domain_name + '_running_var').to(device)
            y = self.bn(x)
        return y

    def update_domain_stats(self, x):
        #implemented from https://doi.org/10.1016/j.patcog.2018.03.005
        device = x.device
        batch_mean = torch.mean(x, dim=[0, 2, 3]).to(device)
        batch_var = torch.var(x, dim=[0, 2, 3], unbiased=True).to(device)
        batch_size = x.shape[0]
        setattr(self, self.current_domain_name + '_num_samples_tracked', getattr(self, self.current_domain_name + '_num_samples_tracked') + batch_size)
        N = getattr(self, self.current_domain_name + '_num_samples_tracked').to(device)
        var = getattr(self, self.current_domain_name + '_running_var').to(device)
        d = batch_mean - getattr(self, self.current_domain_name + '_running_mean').to(device)
        setattr(self, self.current_domain_name + '_running_mean', getattr(self, self.current_domain_name + '_running_mean').to(device) + (d*batch_size)/N)
        setattr(self, self.current_domain_name + '_running_var', (var*N)/(N+batch_size) + (batch_var*batch_size)/(N+batch_size) + (d**2*N*batch_size)/(N+batch_size)**2)


class AdaBatchNorm1d(nn.Module):
    def __init__(self, num_features, domain_name, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps, momentum, affine)
        self.num_features = num_features
        self.current_domain_name = domain_name
        self.register_buffer('source_running_mean', torch.zeros(self.num_features),
                             persistent=True)
        self.register_buffer('source_running_var', torch.ones(self.num_features), persistent=True)
        self.register_buffer('source_num_samples_tracked', torch.zeros(1), persistent=True)

        self.register_buffer('target_running_mean', torch.zeros(self.num_features),
                             persistent=True)
        self.register_buffer('target_running_var', torch.ones(self.num_features), persistent=True)
        self.register_buffer('target_num_samples_tracked', torch.zeros(1), persistent=True)

        self.registered_domains = []

    def forward(self, x):
        batch_size = x.shape[0]
        device = x.device
        if self.current_domain_name not in self.registered_domains:
            self.registered_domains.append(self.current_domain_name)
        if self.training:
            y = self.bn(x)
            if self.current_domain_name == 'source':
                setattr(self, self.current_domain_name + '_running_mean', self.bn.running_mean)
                setattr(self, self.current_domain_name + '_running_var', self.bn.running_var)
                setattr(self, self.current_domain_name + '_num_samples_tracked',
                        torch.tensor([self.bn.num_batches_tracked*batch_size]).to(device))
        else:
            if self.current_domain_name != 'source':
                self.update_domain_stats(x)
            self.bn.running_mean = getattr(self, self.current_domain_name + '_running_mean').to(device)
            self.bn.running_var = getattr(self, self.current_domain_name + '_running_var').to(device)
            y = self.bn(x)
        return y

    def update_domain_stats(self, x):
        # implemented from https://doi.org/10.1016/j.patcog.2018.03.005
        device = x.device
        batch_mean = torch.mean(x, dim=[0]).to(device)
        batch_var = torch.var(x, dim=[0], unbiased=True).to(device)
        batch_size = x.shape[0]
        setattr(self, self.current_domain_name + '_num_samples_tracked',
                getattr(self, self.current_domain_name + '_num_samples_tracked') + batch_size)
        N = getattr(self, self.current_domain_name + '_num_samples_tracked').to(device)
        var = getattr(self, self.current_domain_name + '_running_var').to(device)
        d = batch_mean - getattr(self, self.current_domain_name + '_running_mean').to(device)
        setattr(self, self.current_domain_name + '_running_mean',
                getattr(self, self.current_domain_name + '_running_mean').to(device) + (d * batch_size) / N)
        setattr(self, self.current_domain_name + '_running_var',
                (var * N) / (N + batch_size) + (batch_var * batch_size) / (N + batch_size) + (
                            d ** 2 * N * batch_size) / (N + batch_size) ** 2)
