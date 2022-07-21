from sklearn import svm
from sklearn import metrics
from Dataset.dataset import *

if __name__ == '__main__':
    full_dataset = EEGDataset("./Data/EEGLarge/EEGLarge.npy")
    train_dataset, test_dataset = dataset_split_train_test(full_dataset)

    train_data, train_labels = train_dataset.dataset.data, train_dataset.dataset.labels_class_id
    train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1] * train_data.shape[2]))

    test_data, test_labels = test_dataset.dataset.data, test_dataset.dataset.labels_class_id
    test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1] * test_data.shape[2]))

    clf = svm.SVC(kernel='rbf', C=0.1, cache_size=1800)
    clf.fit(train_data, train_labels)
    label_pred = clf.predict(test_data)

    print("Miara accuracy:", metrics.accuracy_score(test_labels, label_pred))
