from torch.utils.data import DataLoader
from Models.OneDNet import OneDNet
from Dataset.dataset import *
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, StochasticWeightAveraging, ModelCheckpoint

def main():
    included_classes = [0, 1, 2]
    included_channels = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    full_dataset = EEGDataset("../DataBDF/OutKuba/OutKuba_train.npy",
                              "../DataBDF/OutKuba/OutKuba_val.npy",
                              "../DataBDF/OutKuba/OutKuba_test.npy",
                              included_classes, included_channels)
    train_dataset, val_dataset, test_dataset = full_dataset.get_subsets()
    train_data = DataLoader(train_dataset, batch_size=512, shuffle=False, num_workers=8)
    val_data = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_data = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = OneDNet(included_classes, train_indices=train_dataset.indices, val_indices=val_dataset.indices, test_indices=test_dataset.indices)
    sd = model.state_dict()
    model.load_state_dict(sd)
    model = OneDNet.load_from_checkpoint(channel_count=len(included_channels),
                                         included_classes=included_classes,
                                         domain_name='target',
                                         checkpoint_path="../lightning_logs/version_48/checkpoints/epoch=75-val_loss=0.00-val_accuracy=0.00.ckpt")

    trainer = Trainer(accelerator='auto', callbacks=[TQDMProgressBar(refresh_rate=5),
                                          StochasticWeightAveraging(swa_lrs=1e-2),
                                          ModelCheckpoint(filename="{epoch}-{val_loss:.2f}-{val_accuracy:.2f}",
                                                          save_weights_only=False,
                                                          monitor="Val loss",
                                                          save_last=True,
                                                          save_top_k=3,
                                                          mode='min'),
                                          ModelCheckpoint(filename="{epoch}-{val_accuracy:.2f}-{val_loss:.2f}",
                                                          save_weights_only=False,
                                                          monitor="MulticlassAccuracy",
                                                          save_top_k=3,
                                                          mode='max')],
                      check_val_every_n_epoch=2, benchmark=True, max_epochs=1000000)
    for i in range(50):
        trainer.test(model, train_data)

if __name__ == "__main__":
    main()


    # def __init__(self, included_classes, domain_name='source', train_indices=None, val_indices=None, test_indices=None):
    #     super().__init__()
    #     classes_count = len(included_classes)
    #
    #     self.features = nn.Sequential(
    #         nn.Conv2d(1, 1, kernel_size=(5, 64), padding='same', padding_mode='circular'),
    #         nn.AvgPool2d(kernel_size=(1, 3)),
    #         AdaBatchNorm2d(1, domain_name=domain_name),
    #         nn.LeakyReLU(negative_slope=0.05, inplace=True),
    #         #nn.Dropout2d(p=0.7),
    #
    #         nn.Conv2d(1, 2, kernel_size=(5, 32), padding='valid'),
    #         # nn.Dropout2d(p=0.2),
    #         nn.AvgPool2d(kernel_size=(1, 3)),
    #         AdaBatchNorm2d(2, domain_name=domain_name),
    #         nn.LeakyReLU(negative_slope=0.05, inplace=True),
    #         #nn.Dropout2d(p=0.6),
    #
    #         nn.Conv2d(2, 4, kernel_size=(5, 16), padding='same'),
    #         # nn.Dropout2d(p=0.2),
    #         AdaBatchNorm2d(4, domain_name=domain_name),
    #         nn.LeakyReLU(negative_slope=0.05, inplace=True),
    #         #nn.Dropout2d(p=0.6),
    #         nn.Conv2d(4, 8, kernel_size=(3, 8), padding='same'),
    #         # nn.Dropout2d(p=0.2),
    #         AdaBatchNorm2d(8, domain_name=domain_name),
    #         nn.LeakyReLU(negative_slope=0.05, inplace=True),
    #         nn.Conv2d(8, 16, kernel_size=(3, 4), padding='same'),
    #         # nn.Dropout2d(p=0.2),
    #         AdaBatchNorm2d(16, domain_name=domain_name),
    #         nn.LeakyReLU(negative_slope=0.05, inplace=True),
    #
    #         #nn.Dropout2d(p=0.6),
    #     )
    #
    #     self.classifier = nn.Sequential(
    #         nn.Linear(112, 16),
    #         AdaBatchNorm1d(16, domain_name=domain_name),
    #         nn.LeakyReLU(negative_slope=0.05, inplace=True),
    #         #nn.Dropout(p=0.5),
    #
    #         nn.Linear(16, 8),
    #         AdaBatchNorm1d(8, domain_name=domain_name),
    #         nn.LeakyReLU(negative_slope=0.05, inplace=True),
    #         #nn.Dropout(p=0.5),
    #
    #         nn.Linear(8, classes_count),
    #         nn.LogSoftmax(dim=1)
    #     )