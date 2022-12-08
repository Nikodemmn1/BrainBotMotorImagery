import gc
import os
import pickle
import math
import scipy.io
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
# The less threads you use here, the less memory you need.
# If you have a lot of memory, you can also increase it, making the normalization process faster.
# !!!ATTENTION!!!
THREADS_NORMALIZATION = 2
TRAIN_PERCENT = 0.8
VAL_PERCENT = 0.1
TEST_PERCENT = 1 - TRAIN_PERCENT - VAL_PERCENT


def load_mean_std(path):
    with open(path, "rb") as mean_std_file:
        mean_std = pickle.load(mean_std_file)
    return mean_std


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

    def __init__(self, input_folder, output_folder):
        input_paths = [join(input_folder, file_name) for file_name in listdir(input_folder)]
        self.input_file_paths = [input_file_path for input_file_path in input_paths if isfile(input_file_path)]
        self.output_file_base = join(output_folder, basename(normpath(output_folder)))
        self.converted_data = [[], [], []]

        # numerical from 0 to L-1, where L is labels for each dataset:
        self.labels = [[], [], []]

        self.mean_std = None

    def convert_and_save(self):
        self._convert()
        del self.snippets
        self._process_post_convert()
        self._save_data()

    # The unit must be 1μV, self.snippets should be non-overlapping recording fragments of size self.OVERLAP
    def _convert_specific(self, i_file_path):
        raise NotImplementedError

    def _convert(self):
        for i_file_path in self.input_file_paths:
            self.snippets = [[] for _ in range(self.CLASSES_COUNT)]

            self._convert_specific(i_file_path)

            snippets_len_min = min([len(snippet) for snippet in self.snippets])
            train_len = int(np.floor(snippets_len_min * TRAIN_PERCENT))
            val_len = int(np.floor(snippets_len_min * VAL_PERCENT))
            self.snippets = np.array([lc[:snippets_len_min] for lc in self.snippets])
            self.snippets = np.split(self.snippets, [train_len, train_len + val_len], axis=1)

            for dset in range(3):
                for label, labeled_snippets in enumerate(self.snippets[dset]):
                    buffer = np.zeros((len(self.CHANNELS_ORDER), self.BUFFER_LENGTH)).astype('float32')
                    buffer_filled = 0

                    for snippet in labeled_snippets:
                        buffer = np.roll(buffer, -self.OVERLAP, 1)
                        buffer[:, -self.OVERLAP:] = snippet

                        if buffer_filled + self.OVERLAP < self.BUFFER_LENGTH:
                            buffer_filled += self.OVERLAP
                        else:
                            self.converted_data[dset].append(buffer)
                            self.labels[dset].append(label)

    def _filter(self):
        f_ord, wn = buttord(self.LOW_PASS_FREQ_PB, self.LOW_PASS_FREQ_SB, self.MAX_LOSS_PB, self.MIN_ATT_SB, False,
                            self.DATASET_FREQ)
        low_b, low_a, *rest = butter(f_ord, wn, 'lowpass', False, 'ba', self.DATASET_FREQ)

        f_ord, wn = buttord(self.HIGH_PASS_FREQ_PB, self.HIGH_PASS_FREQ_SB, self.MAX_LOSS_PB, self.MIN_ATT_SB, False,
                            self.DATASET_FREQ)
        high_b, high_a, *rest = butter(f_ord, wn, 'highpass', False, 'ba', self.DATASET_FREQ)

        for dset in range(3):
            for sample_no in range(len(self.converted_data[dset])):
                for cl in range(self.converted_data[dset][0].shape[0]):
                    self.converted_data[dset][sample_no][cl] = lfilter(low_b, low_a,
                                                                       self.converted_data[dset][sample_no][cl])
                    self.converted_data[dset][sample_no][cl] = lfilter(high_b, high_a,
                                                                       self.converted_data[dset][sample_no][cl])
                if self.DECIMATION_FACTOR is not None:
                    self.converted_data[dset][sample_no] = np.apply_along_axis(decimate, 1,
                                                                               self.converted_data[dset][sample_no],
                                                                               self.DECIMATION_FACTOR,
                                                                               ftype='fir')

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

    def _normalize(self):
        print("Converting to numpy array...")

        for dset in range(3):
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
            self.converted_data[dset] = (self.converted_data[dset] - mean[None, :, None]) / std[None, :, None]
        self.mean_std = {"mean": mean, "std": std}

    def _order_by_label_and_balance(self):
        labels_unique_values = sorted(list(set(self.labels[0])))

        self.labels = [np.array(labels) for labels in self.labels]

        for dset in range(3):
            self.converted_data[dset] = [self.converted_data[dset][np.where(self.labels[dset] == label)] for label in
                                         labels_unique_values]
            # min_len = min([len(by_label) for by_label in by_label_list])
            # by_label_balance_indices = [np.random.choice(np.arange(len(by_label)), min_len, replace=False)
            #                             for by_label in by_label_list]
            # self.converted_data[dset] = [by_label[by_label_balance_indices[i], :, :] for i, by_label in
            #                              enumerate(by_label_list)]

    def _flatten_data(self):
        self.converted_data = np.array(self.converted_data)
        old_shape = self.converted_data.shape
        new_shape = (old_shape[0], old_shape[1], old_shape[2] * old_shape[3])
        self.converted_data = np.reshape(self.converted_data, new_shape)

    def _process_post_convert(self):
        print("Filtering...")
        self._filter()
        # self._calculate_psd_welch()
        print("Normalizing...")
        self._normalize()
        print("Ordering by label and balancing...")
        self._order_by_label_and_balance()

    def _save_data(self):
        np.save(f"{self.output_file_base}_train.npy", np.array(self.converted_data[0], dtype=np.float32),
                allow_pickle=False, fix_imports=False)
        np.save(f"{self.output_file_base}_val.npy", np.array(self.converted_data[1], dtype=np.float32),
                allow_pickle=False, fix_imports=False)
        np.save(f"{self.output_file_base}_test.npy", np.array(self.converted_data[2], dtype=np.float32),
                allow_pickle=False, fix_imports=False)
        with open(f"{self.output_file_base}_mean_std.txt", 'w') as mean_std_txt_file:
            mean_std_txt_file.write(f"MEAN: {self.mean_std['mean']}\nSTD: {self.mean_std['std']}")
        with open(f"{self.output_file_base}_mean_std.pkl", 'wb') as mean_std_bin_file:
            pickle.dump(self.mean_std, mean_std_bin_file)


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

    DECIMATION_FACTOR = 5

    def _convert_specific(self, i_file_path):
        mat = scipy.io.loadmat(i_file_path)
        raw_data = mat['o']['data'][0, 0][:, self.CHANNELS_ORDER].T.astype('float32')
        markers = mat['o']['marker'][0, 0].flatten()

        slice_start = 0
        curr_marker = -1
        for i in range(markers.size):
            if markers[i] != curr_marker:
                if curr_marker in range(1, 7):
                    recording = raw_data[:, slice_start:i]
                    recording_len_divisible = recording.shape[1] - recording.shape[1] % self.OVERLAP
                    self.snippets[curr_marker - 1] += np.split(recording[:, :recording_len_divisible],
                                                               recording_len_divisible / self.OVERLAP, 1)
                if markers[i] in range(1, 7):
                    slice_start = i
                curr_marker = markers[i]


# TODO: Check trigger values representation in signal on BDF recordings and modify the range(1, 7) accordingly
class BiosemiBDFConverter(EEGDataConverter):
    DATASET_FREQ = 2048

    BUFFER_LENGTH = 8000
    OVERLAP = 800

    CLASSES_COUNT = 3

    LOW_PASS_FREQ_PB = 30
    LOW_PASS_FREQ_SB = 60
    HIGH_PASS_FREQ_PB = 6
    HIGH_PASS_FREQ_SB = 3

    WELCH_OVERLAP_PERCENT = 80
    WELCH_SEGMENT_LEN = 350

    MAX_LOSS_PB = 2
    MIN_ATT_SB = 8

    CHANNELS_IN_FILE = 17  # with triggers
    HEADER_LENGTH = 256 * (CHANNELS_IN_FILE + 1)

    CHANNELS_ORDER = [*range(0, 16, 1)]

    DECIMATION_FACTOR = 50

    def _convert_specific(self, i_file_path):
        file_len_bytes = os.stat(i_file_path).st_size
        file_len_bytes_headless = file_len_bytes - self.HEADER_LENGTH

        channel_sections_count = file_len_bytes_headless // (self.CHANNELS_IN_FILE * self.DATASET_FREQ * 3)

        with open(i_file_path, 'rb') as f:
            data = f.read()
        data = np.frombuffer(data[self.HEADER_LENGTH:], dtype='<u1')

        samples = np.ndarray((self.CHANNELS_IN_FILE - 1,
                              self.DATASET_FREQ * channel_sections_count, 3), dtype='<u1')

        triggers = np.ndarray((1, self.DATASET_FREQ * channel_sections_count, 3), dtype='<u1')

        for sec in range(channel_sections_count):
            for ch in range(self.CHANNELS_IN_FILE):
                for sam in range(self.DATASET_FREQ):
                    beg = sec * self.CHANNELS_IN_FILE * self.DATASET_FREQ * 3 + ch * self.DATASET_FREQ * 3 + sam * 3
                    if ch != self.CHANNELS_IN_FILE - 1:
                        samples[ch, sec * self.DATASET_FREQ + sam, :] = data[beg:beg + 3]
                    else:
                        triggers[0, sec * self.DATASET_FREQ + sam, :] = data[beg:beg + 3]

        raw_data = samples[:, :, 0].astype("int32") + samples[:, :, 1].astype("int32") * 256 + samples[:, :, 2].astype(
            "int32") * 256 * 256
        raw_data[raw_data > pow(2, 23)] -= pow(2, 24)

        markers = triggers[0, :, 1]
        markers = np.where(markers > 0,
                           np.log2(np.bitwise_and(markers, -markers)).astype('int8') + 1,
                           np.zeros(markers.shape[0]).astype('int8'))
        markers -= 1

        raw_data = raw_data.astype('float32')
        raw_data -= 0.55 * (raw_data[6, :] + raw_data[8, :])  # referencing the signal

        slice_start = 0
        curr_marker = -1
        for i in range(markers.size - 1):
            if markers[i] != -1:
                if curr_marker > 0 and markers[i - 1] == -1:
                    recording = raw_data[:, slice_start:i]
                    recording_len_divisible = recording.shape[1] - recording.shape[1] % self.OVERLAP
                    self.snippets[curr_marker - 1] += np.split(recording[:, :recording_len_divisible],
                                                               recording_len_divisible / self.OVERLAP, 1)
                if markers[i] != 0 and markers[i + 1] == -1:
                    slice_start = i
                curr_marker = markers[i]

        for snippet_class in range(self.CLASSES_COUNT):
            for snippet in self.snippets[snippet_class]:
                snippet -= snippet.mean(axis=1)[:, None]
                snippet *= 0.03125

        return
