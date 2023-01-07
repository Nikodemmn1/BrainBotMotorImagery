from Server.server_params import *
import numpy as np

def decode_data_from_bytes(raw_data):
    data_struct = np.zeros((CHANNELS-1, SAMPLES))
    triggers = np.zeros((SAMPLES, 3), dtype='<u1')
    # 32 bit unsigned words reorder
    raw_data_array_base = np.array(raw_data)
    raw_data_array_base = raw_data_array_base.reshape((WORDS, 3))
    raw_data_array = raw_data_array_base.astype("int32")
    raw_data_array = raw_data_array[:, 0].astype("int32") + \
                     raw_data_array[:, 1].astype("int32") * 256 + \
                     raw_data_array[:, 2].astype("int32") * 256 * 256
    raw_data_array[raw_data_array >= (1 << 23)] -= (1 << 24)

    for j in range(CHANNELS - 1):
        for i in range(SAMPLES):
            data_struct[j, i] = raw_data_array[i * CHANNELS + j].astype('float32')

    # setting reference
    data_struct -= 0.55 * (data_struct[6, :] + data_struct[8, :])

    for i in range(SAMPLES):
        triggers[i] = raw_data_array_base[i * CHANNELS + CHANNELS - 1]

    return data_struct, triggers