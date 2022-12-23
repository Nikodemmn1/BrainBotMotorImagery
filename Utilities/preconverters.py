import os
import pickle
from os.path import join, basename, normpath, isfile
import scipy.io
import numpy as np


class PreConverter:
    CLASSES_COUNT = None
    OVERLAP = None

    def __init__(self, input_folder, output_folder):
        input_paths = [join(input_folder, file_name) for file_name in os.listdir(input_folder)]
        self.input_file_paths = [input_file_path for input_file_path in input_paths if isfile(input_file_path)]
        self.output_folder = output_folder
        self.snippets = [[] for _ in range(self.CLASSES_COUNT)]

    def preconvert_file(self, i_file_path):
        raise NotImplementedError()

    def preconvert(self):
        for i_file_path in self.input_file_paths:
            self.preconvert_file(i_file_path)

            #snippets_len_min = min([len(snippet) for snippet in self.snippets])
            #self.snippets = np.array([lc[:snippets_len_min] for lc in self.snippets], dtype='float32')
            out_path = f"{self.output_folder}/{basename(i_file_path)}_snippets.pkl"
            #np.save(out_path, self.snippets, allow_pickle=False, fix_imports=False)

            with open(out_path, 'wb') as handle:
                pickle.dump(self.snippets, handle, protocol=pickle.HIGHEST_PROTOCOL)

            self.snippets = [[] for _ in range(self.CLASSES_COUNT)]



class BiosemiBDFPreConverter(PreConverter):
    DATASET_FREQ = 2048
    OVERLAP = 128
    CLASSES_COUNT = 3
    CHANNELS_IN_FILE = 17  # with triggers
    HEADER_LENGTH = 256 * (CHANNELS_IN_FILE + 1)

    MEAN_PERIOD_LEN = 8192

    def preconvert_file(self, i_file_path):
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

        slice_start = -1
        curr_marker = -1
        for i in range(markers.size - 1):
            if markers[i] != -1:
                if curr_marker > 0 and markers[i - 1] == -1:
                    recording = raw_data[:, slice_start:i]
                    recording_len_excess = recording.shape[1] % self.OVERLAP
                    recording_len_divisible = recording.shape[1] - recording_len_excess
                    recording_splits = np.split(recording[:, recording_len_excess:],
                                  recording_len_divisible / self.OVERLAP, 1)
                    recording_split_indices = np.arange(recording_len_divisible, step=self.OVERLAP)[1:]
                    recording_split_indices = np.append(recording_split_indices, [recording_len_divisible])
                    means = []
                    for split_i in recording_split_indices:
                        raw_data_index = i - recording.shape[1] + recording_len_excess + split_i
                        if raw_data_index < self.MEAN_PERIOD_LEN:
                            mean = raw_data[:, :raw_data_index].mean(axis=1)
                        else:
                            mean = raw_data[:, raw_data_index - self.MEAN_PERIOD_LEN:raw_data_index].mean(axis=1)
                        means.append(mean)
                    self.snippets[curr_marker - 1] += [(
                        recording_splits,
                        means
                    )]
                    slice_start = -1
                if markers[i] != 0 and markers[i + 1] == -1:
                    slice_start = i
                curr_marker = markers[i]


class LargeEEGDataPreConverter(PreConverter):
    CHANNELS_ORDER = [0, 1, 3, 18, 2, 14, 4, 19, 5, 15, 7, 20, 6, 8, 16, 9]
    OVERLAP = 80
    CLASSES_COUNT = 6

    def preconvert_file(self, i_file_path):
        mat = scipy.io.loadmat(i_file_path)
        raw_data = mat['o']['data'][0, 0][:, self.CHANNELS_ORDER].T.astype('float32')
        markers = mat['o']['marker'][0, 0].flatten()

        slice_start = 0
        curr_marker = -1
        for i in range(markers.size):
            if markers[i] != curr_marker:
                if curr_marker in range(1, 7):
                    recording = raw_data[:, slice_start:i]
                    recording_len_excess = recording.shape[1] % self.OVERLAP
                    recording_len_divisible = recording.shape[1] - recording_len_excess
                    beg_excess = recording_len_excess // 2
                    end_excess = recording_len_excess - beg_excess
                    self.snippets[curr_marker - 1] += [np.split(recording[:, beg_excess:-end_excess],
                                                               recording_len_divisible / self.OVERLAP, 1)]
                if markers[i] in range(1, 7):
                    slice_start = i
                curr_marker = markers[i]

        for snippet_class in range(self.CLASSES_COUNT):
            for snippet in self.snippets[snippet_class]:
                snippet_mean = np.concatenate(snippet, axis=1).mean(axis=1)
                for i in range(len(snippet)):
                    snippet[i] -= snippet_mean[:, None]