from xml.etree.ElementInclude import include
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, StochasticWeightAveraging
from Models.MLP import MLP
from Dataset.dataset import *
from Other.joint_regression import JointRegression
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import random
import seaborn as sns
import os.path as path

from SharedParameters.signal_parameters import DATASET_FREQ

def main():
    included_classes = [0, 1]
    freqs = [11, 22]
    m = 10
    included_channels = None #assign None to take all 16 channels
    full_dataset = EEGDataset("./Data/EEGLarge/Train/Train.npy",
                              "./Data/EEGLarge/Val/Val.npy",
                              "./Data/EEGLarge/Test/Test.npy",
                              included_classes=included_classes,
                              included_channels=included_channels,
                              augment=True,
                              augmentations_num=1)
    train_dataset, val_dataset, test_dataset = full_dataset.get_subsets()
    if not path.exists("processed_data_train.npy"):    
        jr = JointRegression(m, len(included_channels) if included_channels != None else 16, freqs, DATASET_FREQ)
        train_indices = train_dataset.indices
        train_data = jr(full_dataset.data[train_indices])
        np.save("processed_data_train.npy", np.append(train_data, full_dataset.labels[train_indices].reshape((-1, 1)), axis = 1), allow_pickle=True)
    if not path.exists("processed_data_val.npy"):    
        jr = JointRegression(m, len(included_channels) if included_channels != None else 16, freqs, DATASET_FREQ)
        val_indices = val_dataset.indices
        val_data = jr(full_dataset.data[val_indices])
        np.save("processed_data_val.npy", np.append(val_data, full_dataset.labels[val_indices].reshape((-1, 1)), axis = 1), allow_pickle=True)
    if not path.exists("processed_data_test.npy"):
        jr = JointRegression(m, len(included_channels) if included_channels != None else 16, freqs, DATASET_FREQ)
        test_indices = test_dataset.indices
        test_data = jr(full_dataset.data[test_indices])
        np.save("processed_data_test.npy", np.append(test_data, full_dataset.labels[test_indices].reshape((-1, 1)), axis = 1), allow_pickle=True)
    
    processed_dataset = processedDataset("./processed_data_train.npy",
                                  "./processed_data_val.npy", 
                                  "./processed_data_test.npy")
    train_dataset, val_dataset, test_dataset = processed_dataset.get_subsets()
    train_data = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=12)
    val_data = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=12)
    test_data = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=12)
    model = MLP(train_dataset.dataset.data.shape[1], len(included_classes), train_indices=train_dataset.indices, val_indices = val_dataset.indices, test_indices = test_dataset.indices)
    trainer = Trainer(accelerator='gpu', devices=-1, callbacks=[TQDMProgressBar(refresh_rate=100), StochasticWeightAveraging(swa_lrs=1e-2)],
                      check_val_every_n_epoch=5, benchmark=True, max_epochs=50)
    trainer.fit(model, train_data, test_data)
    trainer.test(model, test_data)
    #index = random.sample(range(X.shape[0]), 1000 )
    # clf = SVC(gamma='auto', kernel='rbf', verbose=1)
    # clf.fit(X[index], Y[index])
    # y_pred = clf.predict(X)
    # cf = confusion_matrix(Y, y_pred)
    # accuracy = accuracy_score(Y, y_pred)
    # print(cf)
    # print(accuracy)
    # plt.figure(figsize = (10,7))
    # cf_heatmap = sns.heatmap(cf, annot=True)
    # fig = cf_heatmap.get_figure()
    # fig.savefig("confusion_matrix_svm.png") 
if __name__ == "__main__":
    main()
