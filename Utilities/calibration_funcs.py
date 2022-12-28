import numpy as np
def load_calibration_data(data_file, labels_file):
    try:
        data = np.load(data_file)
        labels = np.load(labels_file)
        return data, labels
    except:
        print("Another script is using the files... possibly calibration_server.py saving data.")
        return None