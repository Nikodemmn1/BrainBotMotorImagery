import gc
import pickle
import math
from os import listdir
from os.path import isfile, join, basename, normpath
import numpy as np
from scipy.signal import lfilter, butter, buttord, welch
from scipy.fft import fft, fftfreq
from Utilities.running_stats import RunningStats
from joblib import Parallel, delayed
from scipy.signal import decimate


# !!!ATTENTION!!!
# If you don't have enough memory, change this to something smaller!
# The fewer threads you use here, the less memory you need.
# If you have a lot of memory, you can also increase it, making the normalization process faster.
# !!!ATTENTION!!!
THREADS_NORMALIZATION = 2


def _normalize_job(cl, cl_data):
    print(F"Calculating mean and std for channel {cl}")

    cl_data = cl_data.flatten().astype("float32")

    running_stats = RunningStats()
    push_vectorized = np.vectorize(running_stats.push)

    push_vectorized(cl_data)

    mean = running_stats.mean()
    std = running_stats.standard_deviation()

    return cl, mean, std


class EEGDataConverter:
    DATASET_FREQ = None
    LOW_PASS_FREQ_PB = None
    LOW_PASS_FREQ_SB = None
    HIGH_PASS_FREQ_PB = None
    HIGH_PASS_FREQ_SB = None
    MAX_LOSS_PB = None
    MIN_ATT_SB = None
    CHANNELS_ORDER = None

    WELCH_OVERLAP_PERCENT = None
    WELCH_SEGMENT_LEN = None

    BUFFER_LENGTH = None
    OVERLAP = None

    CLASSES_COUNT = None

    DECIMATION_FACTOR = None

    TRAIN_PERCENT = 0.8
    VAL_PERCENT = 0.1
    TEST_PERCENT = 1 - TRAIN_PERCENT - VAL_PERCENT

    def __init__(self, input_folders, output_folder):
        input_paths = [[join(i_folder, file_name) for file_name in listdir(i_folder)] for i_folder in input_folders]
        self.input_file_paths = [[i for i in input_paths_d if isfile(i)] for input_paths_d in input_paths]
        self.output_file_base = join(output_folder, basename(normpath(output_folder)))
        self.converted_data = [[], [], []]

        # numerical from 0 to L-1, where L is labels for each dataset:
        self.labels = [[], [], []]

        self.mean_std = None

        f_ord, wn = buttord(self.LOW_PASS_FREQ_PB, self.LOW_PASS_FREQ_SB, self.MAX_LOSS_PB, self.MIN_ATT_SB, False,
                            self.DATASET_FREQ)
        self.low_b, self.low_a, *rest = butter(f_ord, wn, 'lowpass', False, 'ba', self.DATASET_FREQ)

        f_ord, wn = buttord(self.HIGH_PASS_FREQ_PB, self.HIGH_PASS_FREQ_SB, self.MAX_LOSS_PB, self.MIN_ATT_SB, False,
                            self.DATASET_FREQ)
        self.high_b, self.high_a, *rest = butter(f_ord, wn, 'highpass', False, 'ba', self.DATASET_FREQ)

    def convert_and_save(self):
        self._convert()
        self._process_post_convert()
        self._save_data()

    def _convert(self):
        #in_files_count = len(self.input_file_paths)
        #train_len = int(np.floor(in_files_count * self.TRAIN_PERCENT))
        #val_len = int(np.floor(in_files_count * self.VAL_PERCENT))
        #all_file_indices = np.arange(in_files_count)
        #np.random.shuffle(all_file_indices)
        #file_indices = np.split(all_file_indices, [train_len, train_len + val_len])
        buffer_duplicate = np.zeros((len(self.CHANNELS_ORDER), self.BUFFER_LENGTH)).astype('float32')

        for dset in range(3):
            for i_file_path in self.input_file_paths[dset]:
                with open(i_file_path, 'rb') as handle:
                    snippets = pickle.load(handle)
                for label, labeled_snippets in enumerate(snippets):
                    for snippet in labeled_snippets:
                        buffer = np.zeros((len(self.CHANNELS_ORDER), self.BUFFER_LENGTH)).astype('float32')
                        buffer_filled = 0
                        assert len(snippet[0]) == len(snippet[1])
                        for mini_snippet_i in range(len(snippet[0])):
                            mini_snippet = snippet[0][mini_snippet_i]
                            mini_snippet_mean = snippet[1][mini_snippet_i][:, None]
                            buffer = np.roll(buffer, -self.OVERLAP, 1)
                            buffer[:, -self.OVERLAP:] = mini_snippet

                            if buffer_filled + self.OVERLAP < self.BUFFER_LENGTH:
                                buffer_filled += self.OVERLAP
                            else:
                                buffer_duplicate[:, ...] = buffer
                                buffer -= mini_snippet_mean
                                buffer *= 0.03125
                                buffer = self._filter(buffer)
                                buffer -= buffer.min(axis=1)[:, None]
                                buffer += 5e-10
                                buffer = np.log(buffer)
                                self.converted_data[dset].append(buffer)
                                self.labels[dset].append(label)
                                buffer = buffer_duplicate
                del snippets
                gc.collect()

    def _process_post_convert(self):
        print("Normalizing...")
        self._normalize()
        print("Ordering by label and balancing...")
        self._order_by_label_and_balance()

    def _filter(self, buffer):
        for cl in range(buffer.shape[0]):
            buffer[cl] = lfilter(self.low_b, self.low_a, buffer[cl])
            buffer[cl] = lfilter(self.high_b, self.high_a, buffer[cl])
        if self.DECIMATION_FACTOR is not None:
            buffer = np.apply_along_axis(decimate, 1, buffer, self.DECIMATION_FACTOR, ftype='fir')
        return buffer

    def _normalize(self):
        print("Converting to numpy array...")

        for dset in range(3):
            if len(self.converted_data[dset]) > 0:
                self.converted_data[dset] = np.dstack(self.converted_data[dset]).astype('float32').transpose((2, 0, 1))
                gc.collect()

        print("Convertion complete, proceeding with normalization...")

        channel_count = self.converted_data[0].shape[1]

        results = Parallel(n_jobs=THREADS_NORMALIZATION)(delayed(_normalize_job)(cl,
                                                                                 self.converted_data[0][:, cl,
                                                                                 :])
                                                         for cl in range(channel_count))

        results = sorted(results, key=lambda x: x[0])
        results = list(map(lambda x: np.asarray([x[1], x[2]]).astype('float32'), results))
        results = np.vstack(results).astype('float32')

        mean = results[:, 0]
        std = results[:, 1]

        std = np.where(std == 0.0, np.ones(std.shape[0]), std).astype('float32')

        print(mean)
        print(std)

        self._normalize_post_calc(mean, std)

    def _normalize_post_calc(self, mean, std):
        for dset in range(3):
            if len(self.converted_data[dset]) > 0:
                self.converted_data[dset] = (self.converted_data[dset] - mean[None, :, None]) / std[None, :, None]
        self.mean_std = {"mean": mean, "std": std}

    def _order_by_label_and_balance(self):
        labels_unique_values = sorted(list(set(self.labels[0])))

        self.labels = [np.array(labels) for labels in self.labels]

        for dset in range(3):
            if len(self.converted_data[dset]) > 0:
                self.converted_data[dset] = [self.converted_data[dset][np.where(self.labels[dset] == label)]
                                             for label in labels_unique_values]
                min_len = min([label_data.shape[0] for label_data in self.converted_data[dset]])
                self.converted_data[dset] = [label_data[:min_len, ...] for label_data in self.converted_data[dset]]

    def _save_data(self):
        datasets_names = ['train', 'val', 'test']
        for dset in range(3):
            np.save(f"{self.output_file_base}_{datasets_names[dset]}.npy", np.array(self.converted_data[dset],
                    dtype=np.float32), allow_pickle=False, fix_imports=False)
        with open(f"{self.output_file_base}_mean_std.txt", 'w') as mean_std_txt_file:
            mean_std_txt_file.write(f"MEAN: {self.mean_std['mean']}\nSTD: {self.mean_std['std']}")
        with open(f"{self.output_file_base}_mean_std.pkl", 'wb') as mean_std_bin_file:
            pickle.dump(self.mean_std, mean_std_bin_file)

        # It also performs filtering!

    def _calculate_psd_fft(self):
        self.converted_data = np.apply_along_axis(self._calculate_psd_fft_channel, 1, self.converted_data)

    def _calculate_psd_fft_channel(self, c):
        densities = np.abs(fft(c))
        freqs = fftfreq(len(densities), 1 / self.DATASET_FREQ)
        return self._filter_densities(densities, freqs)

        # It also performs filtering!

    def _calculate_psd_welch(self):
        self.converted_data = [np.apply_along_axis(self._calculate_psd_welch_channel, 1, recording) for recording in
                               self.converted_data]

    def _calculate_psd_welch_channel(self, c):
        welch_overlap_len = math.ceil((self.WELCH_OVERLAP_PERCENT / 100.0) * self.WELCH_SEGMENT_LEN)
        freqs, densities = welch(c, self.DATASET_FREQ, nperseg=self.WELCH_SEGMENT_LEN, noverlap=welch_overlap_len,
                                 scaling='spectrum', detrend=False)
        return self._filter_densities(densities, freqs)

    def _filter_densities(self, densities, freqs):
        filter_indices = np.intersect1d(np.where(freqs > self.LOW_PASS_FREQ_PB),
                                        np.where(freqs < self.HIGH_PASS_FREQ_PB))
        return densities[filter_indices]


class LargeEEGDataConverter(EEGDataConverter):
    DATASET_FREQ = 200
    CHANNELS_ORDER = [0, 1, 3, 18, 2, 14, 4, 19, 5, 15, 7, 20, 6, 8, 16, 9]

    BUFFER_LENGTH = 800
    OVERLAP = 80

    CLASSES_COUNT = 6

    LOW_PASS_FREQ_PB = 30
    LOW_PASS_FREQ_SB = 60
    HIGH_PASS_FREQ_PB = 6
    HIGH_PASS_FREQ_SB = 3

    WELCH_OVERLAP_PERCENT = 80
    WELCH_SEGMENT_LEN = 350

    MAX_LOSS_PB = 2
    MIN_ATT_SB = 8

    DECIMATION_FACTOR = 4


class BiosemiBDFConverter(EEGDataConverter):
    DATASET_FREQ = 2048

    BUFFER_LENGTH = 3200
    OVERLAP = 128

    CLASSES_COUNT = 3

    LOW_PASS_FREQ_PB = 30
    LOW_PASS_FREQ_SB = 60
    HIGH_PASS_FREQ_PB = 6
    HIGH_PASS_FREQ_SB = 3

    WELCH_OVERLAP_PERCENT = 80
    WELCH_SEGMENT_LEN = 350

    MAX_LOSS_PB = 2
    MIN_ATT_SB = 6

    CHANNELS_IN_FILE = 17  # with triggers
    HEADER_LENGTH = 256 * (CHANNELS_IN_FILE + 1)

    CHANNELS_ORDER = [*range(0, 16, 1)]

    DECIMATION_FACTOR = 30
