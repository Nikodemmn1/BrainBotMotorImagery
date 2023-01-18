from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from Models.OneDNet import OneDNet
from Dataset.dataset import *
import torch
#from Utilities.test_merger import TestMerger


def main():
    #test_merger = TestMerger("./DataTest/", "./DataTest/", 'Out')
    #test_merger.merge()Olek
    included_classes = [0, 1, 2]
    included_channels = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    full_dataset = EEGDataset("./DataBDF/OutOlek/OutOlek_test.npy",
                              "./DataBDF/OutOlek/OutOlek_test.npy",
                              "./DataBDF/OutOlek/OutOlek_test.npy",
                              included_classes, included_channels)
    # full_dataset = EEGDataset("./DataBDF/Out/Out_train.npy",
    #                           "./DataBDF/Out/Out_val.npy",
    #                           "./DataBDF/Out/Out_test.npy",
    #                          included_classes, included_channels)
    _, _, test_dataset = full_dataset.get_subsets()
    test_data = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    model = OneDNet.load_from_checkpoint(channel_count=len(included_channels),
                                         included_classes=included_classes,
                                         checkpoint_path="./model_for_test.ckpt")
    # model = torch.load("./model_nikodem.pt")
    trainer = Trainer(gpus=-1)
    trainer.test(model, test_data)


if __name__ == "__main__":
    main()
