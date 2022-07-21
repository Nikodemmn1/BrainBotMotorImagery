import pickle
import math
import mne
import scipy.io
from os import listdir
from os.path import isfile, join, basename, normpath
import numpy as np
from scipy.signal import lfilter, butter, buttord, welch
from scipy.fft import fft, fftfreq


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

    def __init__(self, input_folder, output_folder):
        input_paths = [join(input_folder, file_name) for file_name in listdir(input_folder)]
        self.input_file_paths = [input_file_path for input_file_path in input_paths if isfile(input_file_path)]
        self.output_file_base = join(output_folder, basename(normpath(output_folder)))
        self.converted_data = []
        self.labels = []  # numerical from 0 to L-1, where L is labels count
        self.mean_std = None

    def convert_and_save(self):
        self._convert()
        self._process_post_convert()
        self._save_data()

    # The unit must be 1μV, self.snippets should be non-overlapping recording fragments of size self.OVERLAP
    def _convert_specific(self, i_file_path):
        raise NotImplementedError

    def _convert(self):
        for i_file_path in self.input_file_paths:
            self.snippets = [[] for _ in range(self.CLASSES_COUNT)]

            self._convert_specific(i_file_path)

            for label, labeled_snippets in enumerate(self.snippets):
                buffer = np.zeros((len(self.CHANNELS_ORDER), self.BUFFER_LENGTH))
                buffer_filled = 0

                for snippet in labeled_snippets:
                    buffer = np.roll(buffer, -self.OVERLAP, 1)
                    buffer[:, -self.OVERLAP:] = snippet

                    if buffer_filled + self.OVERLAP < self.BUFFER_LENGTH:
                        buffer_filled += self.OVERLAP
                    else:
                        self.converted_data.append(buffer)
                        self.labels.append(label)

    def _filter(self):
        f_ord, wn = buttord(self.LOW_PASS_FREQ_PB, self.LOW_PASS_FREQ_SB, self.MAX_LOSS_PB, self.MIN_ATT_SB, False,
                            self.DATASET_FREQ)
        low_b, low_a, *rest = butter(f_ord, wn, 'lowpass', False, 'ba', self.DATASET_FREQ)

        f_ord, wn = buttord(self.HIGH_PASS_FREQ_PB, self.HIGH_PASS_FREQ_SB, self.MAX_LOSS_PB, self.MIN_ATT_SB, False,
                            self.DATASET_FREQ)
        high_b, high_a, *rest = butter(f_ord, wn, 'highpass', False, 'ba', self.DATASET_FREQ)

        self.converted_data = np.apply_along_axis(lambda c: lfilter(low_b, low_a, c), 1, self.converted_data)
        self.converted_data = np.apply_along_axis(lambda c: lfilter(high_b, high_a, c), 1, self.converted_data)

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
        all_data = np.hstack(self.converted_data)
        mean = np.apply_along_axis(np.mean, 1, all_data)
        std = np.apply_along_axis(np.std, 1, all_data)
        self.converted_data = [np.apply_along_axis(lambda c: (c - mean) / std, 0, c_data)
                               for c_data in self.converted_data]
        self.mean_std = {"mean": mean, "std": std}

    def _order_by_label_and_balance(self):
        labels_unique_values = sorted(list(set(self.labels)))
        self.converted_data = np.array(self.converted_data)
        self.labels = np.array(self.labels)
        by_label_list = [self.converted_data[np.where(self.labels == label)] for label in labels_unique_values]
        min_len = min([len(by_label) for by_label in by_label_list])
        by_label_balance_indices = [np.random.choice(np.arange(len(by_label)), min_len, replace=False)
                                    for by_label in by_label_list]
        self.converted_data = [by_label[by_label_balance_indices[i], :, :] for i, by_label in
                               enumerate(by_label_list)]

    def _flatten_data(self):
        self.converted_data = np.array(self.converted_data)
        old_shape = self.converted_data.shape
        new_shape = (old_shape[0], old_shape[1], old_shape[2] * old_shape[3])
        self.converted_data = np.reshape(self.converted_data, new_shape)

    def _process_post_convert(self):
        self._calculate_psd_welch()
        self._normalize()
        self._order_by_label_and_balance()

    def _save_data(self):
        np.save(f"{self.output_file_base}.npy", np.array(self.converted_data, dtype=np.float32),
                allow_pickle=False, fix_imports=False)
        with open(f"{self.output_file_base}_mean_std.txt", 'w') as mean_std_txt_file:
            mean_std_txt_file.write(f"MEAN: {self.mean_std['mean']}\nSTD: {self.mean_std['std']}")
        with open(f"{self.output_file_base}_mean_std.pkl", 'wb') as mean_std_bin_file:
            pickle.dump(self.mean_std, mean_std_bin_file)


class LargeEEGDataConverter(EEGDataConverter):
    DATASET_FREQ = 200
    LOW_PASS_FREQ_PB = 0.53
    HIGH_PASS_FREQ_PB = 70
    CHANNELS_ORDER = [0, 1, 3, 18, 2, 14, 4, 19, 5, 15, 7, 20, 6, 8, 16, 9]
    WELCH_OVERLAP_PERCENT = 80
    WELCH_SEGMENT_LEN = 350

    BUFFER_LENGTH = 800
    OVERLAP = 80

    CLASSES_COUNT = 6

    def _convert_specific(self, i_file_path):
        mat = scipy.io.loadmat(i_file_path)
        raw_data = mat['o']['data'][0, 0][:, self.CHANNELS_ORDER].T
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
    DATASET_FREQ = 2000
    LOW_PASS_FREQ_PB = 0.53
    HIGH_PASS_FREQ_PB = 70
    CHANNELS_ORDER = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    WELCH_OVERLAP_PERCENT = 80
    WELCH_SEGMENT_LEN = 350

    BUFFER_LENGTH = 8000
    OVERLAP = 800

    CLASSES_COUNT = 6

    def __init__(self, input_folder, output_folder, decimate=False):
        super().__init__(input_folder, output_folder)
        self.decimate = decimate

    def _convert_specific(self, i_file_path):
        bdf_data = mne.io.read_raw_bdf(i_file_path, verbose=False)
        raw_data = bdf_data.get_data()
        triggers = raw_data[-1, :]
        data = raw_data[self.CHANNELS_ORDER, :].astype(np.float32)
        data *= 1e6

        slice_start = 0
        curr_trigger = -1
        for i in range(triggers.size):
            if triggers[i] != curr_trigger:
                if curr_trigger in range(1, 7):
                    recording = data[:, slice_start:i]
                    recording_len_divisible = recording.shape[1] - recording.shape[1] % self.OVERLAP
                    self.snippets[curr_trigger - 1] += np.split(recording[:, :recording_len_divisible],
                                                                recording_len_divisible / self.OVERLAP, 1)
                if triggers[i] in range(1, 7):
                    slice_start = i
                curr_trigger = triggers[i]
