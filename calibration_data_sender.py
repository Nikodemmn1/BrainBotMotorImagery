import numpy as np
import time
from Dataset.dataset import EEGDataset

SAMPLE_TIME = 0.0625
PACKETS_IN_BUFFOR = 16
TRAINING_DATA_PATH = "./DataBDF/OutTraining/Nikodem/Nikodem_train.npy"
SLEEP_TIME = PACKETS_IN_BUFFOR * SAMPLE_TIME
def main():
    included_classes = [0, 1, 2]
    # included_channels = range(16)
    included_channels = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    full_dataset = EEGDataset("./DataBDF/OutTraining/Piotr/Piotr_train.npy",
                              "./DataBDF/OutTraining/Piotr/Piotr_val.npy",
                              "./DataBDF/OutTesting/Piotr/Piotr_test.npy",
                              included_classes, included_channels)
    train_dataset, val_dataset, test_dataset = full_dataset.get_subsets()
    data = train_dataset.dataset.data
    labels = train_dataset.dataset.labels
    counter = 0
    while True:
        counter += 1
        print("Saving data..")
        np.save("CalibrationData/calibration_data_nikodem.npy", data[:16 * counter])
        np.save("CalibrationData/calibration_labels_nikodem.npy", labels[:16 * counter])
        time.sleep(SLEEP_TIME)

if __name__ == "__main__":
    main()