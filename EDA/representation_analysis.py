import numpy as np
from Models.OneDNet import OneDNet, FeatureExtractorOneDNet
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from Dataset.dataset import *
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def tsne(features_vector, labels):
    X = TSNE(n_components=2, learning_rate='auto',
                init='random', perplexity=3).fit_transform(features_vector)
    fig = plt.figure()
    y = np.choose(labels, [0, 1, 2]).astype(float)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.nipy_spectral, edgecolor="k")
    plt.savefig("tsne.png")
    plt.close()
    return

def pca(features_vector, labels):
    pca = PCA(n_components=20)
    pca.fit(features_vector)
    X = pca.transform(features_vector)
    exp_var_pca = pca.explained_variance_ratio_
    fig = plt.figure()
    plt.bar(range(0, len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
    plt.savefig("explained_variance.png")
    plt.close()
    fig = plt.figure()
    y = np.choose(y, [0, 1, 2]).astype(float)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.nipy_spectral, edgecolor="k")
    plt.xlim([-100, 100])
    plt.ylim([-100, 100])
    plt.savefig("pca.jpg")
    plt.close()
    return

def gaussian_kernel(x, y, sigma=1):
    gauss = np.exp(-(np.square(x) + np.square(y))/(2*np.square(sigma)))
    gauss = 1/(2*np.pi*np.square(sigma))*gauss
    return gauss

def calculate_MMD(x, y):
    x_len = x.shape[0]
    y_len = y.shape[0]
    component1 = 0
    component2 = 0
    component3 = 0
    for i in range(x_len):
        for j in range(x_len):
            component1 += gaussian_kernel(x[i], x[j])
    for i in range(y_len):
        for j in range(y_len):
            component2 += gaussian_kernel(y[i], y[j])
    for i in range(x_len):
        for j in range(y_len):
            component3 += gaussian_kernel(x[i], y[j])
    mmd = (1/np.square(x_len))*component1 + (1/np.square(y_len))*component2 - (2/(x_len*y_len))*component3
    return mmd


def main():
    included_classes = [0, 1, 2]
    included_channels = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # included_channels = range(16)
    full_dataset = EEGDataset("../DataBDF/OutTraining/Nikodem/Nikodem_train.npy",
                              "../DataBDF/OutTraining/Nikodem/Nikodem_val.npy",
                              "../DataBDF/OutTesting/Nikodem/Nikodem_test.npy",
                              included_classes, included_channels)
    train_dataset, val_dataset, test_dataset = full_dataset.get_subsets()
    train_data = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_data = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_data = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = OneDNet.load_from_checkpoint(channel_count=len(included_channels),
                                         included_classes=included_classes,
                                         checkpoint_path="../lightning_logs/Nikodem model/checkpoints/epoch=63-step=11072.ckpt")
    data = torch.from_numpy(train_dataset[:][0])
    y = train_dataset[:][1]
    embedding = model.extract_features(data[:])
    embedding = embedding.detach().numpy()
    # representation2 = np.random.normal(0, 1, size=representation.shape)
    # mmd = calculate_MMD(representation, representation2)
    # print(max(mmd)*10000)
    tsne(embedding, y)
if __name__ == "__main__":
    main()