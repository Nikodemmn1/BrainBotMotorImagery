from Models.OneDNet import *
from Dataset.dataset import EEGDataset
from torch.utils.data import DataLoader
import torch.cuda
import numpy as np
from pytorch_lightning.callbacks import TQDMProgressBar, StochasticWeightAveraging, ModelCheckpoint
from tqdm import tqdm

def standarize_features(features):
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    features_std = (features - mean)/std
    return features_std, mean, std

def PCA(features, pc_num):
    features_std, mean, std = standarize_features(features)
    cov_mat = np.cov(features_std.T)
    eigen_vals, eigen_vecs = np.linalg(cov_mat)
    return eigen_vals, eigen_vecs

class OneDNetFeatures(OneDNet):
    def __init(self):
        super().__init__()
    def forward(self, x):
        x = self.features(x)
        features = torch.flatten(x, 1)
        return features

def main():
    torch.cuda.empty_cache()
    included_classes = [0, 1, 2]
    # included_channels = [3, 6, 7, 8, 11]
    included_channels = range(16)
    full_dataset = EEGDataset("../Data/EEGLarge/EEGLarge_train.npy",
                              "../Data/EEGLarge/EEGLarge_val.npy",
                              "../Data/EEGLarge/EEGLarge_test.npy",
                              included_classes, included_channels)
    train_dataset, val_dataset, test_dataset = full_dataset.get_subsets()
    train_data = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1)
    val_data = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_data = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = OneDNetFeatures.load_from_checkpoint(channel_count=len(included_channels),
                                                 included_classes=included_classes,
                                                 checkpoint_path="../model.ckpt")
    model.eval()

    batch_size = 32
    batches_data = []
    batches_labels = []
    for i in tqdm(range(train_dataset.dataset.data.shape[0] // batch_size)):
        sample_data, sample_y = train_dataset.__getitem__(slice(i*batch_size, (i+1)*batch_size))
        batches_data.append(sample_data)
        batches_labels.append(sample_y)
    features = model(torch.from_numpy(batches_data[0])).detach().numpy()
    eigen_vals, eigen_vecs = PCA(features, 2)
    print(eigen_vals)


if __name__ == "__main__":
    main()
