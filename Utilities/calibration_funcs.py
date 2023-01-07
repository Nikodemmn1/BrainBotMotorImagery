import numpy as np
def load_calibration_data(data_file, labels_file):
    try:
        raw_data = np.load(data_file).astype(np.float32)
        labels = np.load(labels_file, allow_pickle=True)
        min_length = np.min([raw_data.shape[0], labels.shape[0]])
        return raw_data[:min_length], labels[:min_length]
    except:
        print("Another script is using the files... possibly calibration_server.py saving data.")
        return None

