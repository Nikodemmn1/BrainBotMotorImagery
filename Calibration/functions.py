from torch.utils.data import DataLoader
from Dataset.dataset import *

def get_dataloaders(dataset_path_prefix, included_classes, included_channels):

    source_dataset = EEGDataset(dataset_path_prefix + "_train.npy",
                                dataset_path_prefix + "_val.npy",
                                dataset_path_prefix + "_test.npy",
                                included_classes, included_channels)

    train_dataset, val_dataset, test_dataset = source_dataset.get_subsets()

    train_loader = DataLoader(train_dataset, batch_size=32, num_workers=12)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=0, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False)

    return {"train_dataloader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader}
