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

class StreamingMeanStd():
    def __init__(self, mean_init, std=0):
        'mean_init should be initialize using first sample'
        self.mean = mean_init
        self.std = std
        self.samples = 1
        self.v = 0
    def __call__(self, x):
        # https://math.stackexchange.com/questions/20593/calculate-variance-from-a-stream-of-sample-values
        self.samples = self.samples + 1
        next_m = self.mean + (x - self.mean)/self.samples
        next_v = self.v + (x - self.mean)*(x - next_m)
        self.v = next_v
        self.mean = next_m
        self.std = np.sqrt(self.v/(self.samples - 1))