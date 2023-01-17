import time
import os
import numpy as np
import matplotlib.pyplot as plt
from Server.server_params import *
import socket
from tqdm import tqdm

LOAD = False

CHANNELS_TO_SEND = CHANNELS

if not LOAD:
    # https://www.biosemi.com/faq/file_format.htm
    CHANNELS_IN_FILE = 17  # with triggers
    HEADER_LENGTH = 256 * (CHANNELS_IN_FILE + 1)
    SAMPLING_RATE = 2048
    DATA_PATH = "DataBDF/Piotr/"
    FILE_PATHS = os.listdir(DATA_PATH)
    samples_list = []
    for file_path in FILE_PATHS:
        path = DATA_PATH + file_path
        file_bytes = os.stat(path).st_size
        file_bytes_no_head = file_bytes - HEADER_LENGTH

        channel_sections_count = file_bytes_no_head // (CHANNELS_IN_FILE * SAMPLING_RATE * 3)

        with open(path, 'rb') as f:
            data = f.read()
        data = np.frombuffer(data[HEADER_LENGTH:], dtype='<u1')

        samples = np.ndarray((CHANNELS_TO_SEND, SAMPLING_RATE * channel_sections_count, 3), dtype='<u1')

        for sec in tqdm(range(channel_sections_count)):
            for ch in range(CHANNELS_TO_SEND):
                for sam in range(SAMPLING_RATE):
                    beg = sec * CHANNELS_IN_FILE * SAMPLING_RATE * 3 + ch * SAMPLING_RATE * 3 + sam * 3
                    samples[ch, sec * SAMPLING_RATE + sam, :] = data[beg:beg + 3]
        samples_list.append(samples)
    samples_to_save = np.concatenate(samples_list, axis = 1)
    np.save("testdata_piotr.npy", samples_to_save)
else:
    samples = np.load("testdata_piotr.npy")

#samples2 = samples[:, :, 0].astype("int32") + samples[:, :, 1].astype("int32") * 256 + samples[:, :, 2].astype(
#    "int32") * 256 * 256
#samples2[samples2 > pow(2, 23)] -= pow(2, 24)

samples = np.transpose(samples, (1, 0, 2)).flatten()
packets_data = np.reshape(samples, (-1, WORDS * 3))

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(("", TCP_AV_PORT))
sock.listen(0)
conn, addr = sock.accept()

with conn:
    print("Connected!")
    for i in range(packets_data.shape[0]):
        conn.send(packets_data[i, :].tobytes())
        time.sleep(0.0625)
