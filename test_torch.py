from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from Models.OneDNet import OneDNet
from Dataset.dataset import *
#from Utilities.test_merger import TestMerger


def main():
    #test_merger = TestMerger("./DataTest/", "./DataTest/", 'Out')
    #test_merger.merge()
    included_classes = [0, 1, 2]
    included_channels = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    full_dataset = EEGDataset("./DataBDF/Out/Out_train.npy",
                              "./DataBDF/Out/Out_val.npy",
                              "./DataBDF/Out/Out_test.npy",
                              included_classes, included_channels)
    _, _, test_dataset = full_dataset.get_subsets()
    test_data = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    model = OneDNet.load_from_checkpoint(channel_count=len(included_channels),
                                         included_classes=included_classes,
                                         checkpoint_path="./Calibration/lightning_logs/version_78/checkpoints/epoch=459-step=19159.ckpt")
    trainer = Trainer(gpus=-1)
    trainer.test(model, test_data)


if __name__ == "__main__":
    main()
