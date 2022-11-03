import pywt
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import lfilter, butter, buttord

from SharedParameters.signal_parameters import CAL, OFFSET, UNIT, LOW_PASS_FREQ_PB, \
    HIGH_PASS_FREQ_PB, WELCH_OVERLAP_PERCENT, WELCH_SEGMENT_LEN, LOW_PASS_FREQ_SB,  \
    MAX_LOSS_PB, MIN_ATT_SB, HIGH_PASS_FREQ_SB
from Server.server_params import *
from scipy.signal import welch, decimate


def eeg_signal_to_dwt(data):
    c_allchannels = np.empty(0)
    for channel in data:
        ca1, cd1 = pywt.dwt(channel, 'db1')
        c_allchannels = np.append(c_allchannels, ca1)
        c_allchannels = np.append(c_allchannels, cd1)
    return c_allchannels


def decode_data_from_bytes(raw_data):
    data_struct = np.zeros((CHANNELS, SAMPLES))

    # 32 bit unsigned words reorder
    raw_data_array = np.array(raw_data)
    raw_data_array = raw_data_array.reshape((WORDS, 3))
    raw_data_array = raw_data_array.astype("int32")
    raw_data_array = ((raw_data_array[:, 0]) +
                      (raw_data_array[:, 1] << 8) +
                      (raw_data_array[:, 2] << 16))
    raw_data_array[raw_data_array >= (1 << 23)] -= (1 << 24)

    for j in range(CHANNELS):
        for i in range(SAMPLES):
            data_struct[j, i] = raw_data_array[i * CHANNELS + j].astype('float64')
            #data_struct[j, i] *= CAL
            #data_struct[j, i] += OFFSET
            #data_struct[j, i] *= UNIT

    return data_struct


def _filter(data):
    f_ord, wn = buttord(LOW_PASS_FREQ_PB, LOW_PASS_FREQ_SB, MAX_LOSS_PB, MIN_ATT_SB, False,
                        BIOSEMI_FREQ)
    low_b, low_a, *rest = butter(f_ord, wn, 'lowpass', False, 'ba', BIOSEMI_FREQ)

    f_ord, wn = buttord(HIGH_PASS_FREQ_PB, HIGH_PASS_FREQ_SB, MAX_LOSS_PB, MIN_ATT_SB, False,
                        BIOSEMI_FREQ)
    high_b, high_a, *rest = butter(f_ord, wn, 'highpass', False, 'ba', BIOSEMI_FREQ)

    converted_data = np.apply_along_axis(lambda c: lfilter(low_b, low_a, c), 1, data)
    converted_data = np.apply_along_axis(lambda c: lfilter(high_b, high_a, c), 1, converted_data)

    return converted_data


def set_reference(data):
    # Reference is 0.55*(C3 + C4) - C3 is channel 6, C4 is channel 8.
    return data - (0.55 * (data[6, :] + data[8, :]))


def calculate_psd_welch_channel(c):
    welch_overlap_len = math.ceil((WELCH_OVERLAP_PERCENT / 100.0) * WELCH_SEGMENT_LEN)
    freqs, densities = welch(c, DATASET_FREQ, nperseg=WELCH_SEGMENT_LEN, noverlap=welch_overlap_len,
                             scaling='spectrum', detrend=False)
    filter_indices = np.intersect1d(np.where(freqs > LOW_PASS_FREQ_PB),
                                    np.where(freqs < HIGH_PASS_FREQ_PB))
    return densities[filter_indices]


def prepare_data_for_classification(data, mean, std):
    # data = data[:CHANNELS-1, :]

    data_with_reference = set_reference(data)

    data_filtered = _filter(data_with_reference)
    data_filtered = data_filtered[:, (data_filtered.shape[1] - 800 * DECIMATION_FACTOR):]

    data_decimated = np.apply_along_axis(decimate, 1, data_filtered, int(DECIMATION_FACTOR))

    # data_psd = np.apply_along_axis(calculate_psd_welch_channel, 1, data_scaled)
    # data_psd[:, 0] = np.zeros(16)

    mean = np.expand_dims(mean, 1)
    std = np.expand_dims(std, 1)
    data_normalized = (data_decimated - mean) / std

    data_reshaped = np.reshape(data_normalized, (1, 1, data_normalized.shape[0], data_normalized.shape[1]))

    return torch.from_numpy(data_reshaped).type(torch.float32)


def get_classification(x, model):
    with torch.no_grad():
        return model(x)
