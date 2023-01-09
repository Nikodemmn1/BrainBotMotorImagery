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
    def __init__(self, x0):
        if not isinstance(x0, np.ndarray):
            x0 = np.array(x0)
        x0 = np.squeeze(x0)
        self.samples = 1
        self.mean = x0.transpose()[0]
        self.v = np.zeros(self.mean.shape)
        for i, sample in enumerate(x0.transpose()[1:]):
            self.samples += 1
            next_m = self.mean + (sample - self.mean) / self.samples
            next_v = self.v + (sample - self.mean) * (sample - next_m)
            self.v = next_v
            self.mean = next_m
        self.std = np.sqrt(self.v/(self.samples - 1))
    def __call__(self, x):
        #https://math.stackexchange.com/questions/20593/calculate-variance-from-a-stream-of-sample-values
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        x = np.squeeze(x)
        for i, sample in enumerate(x.transpose()):
            self.samples += 1
            next_m = self.mean + (sample - self.mean) / self.samples
            next_v = self.v + (sample - self.mean) * (sample - next_m)
            self.v = next_v
            self.mean = next_m
        self.std = np.sqrt(self.v / (self.samples - 1))