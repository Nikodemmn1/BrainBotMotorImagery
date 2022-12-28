import time
import os
import numpy as np
import matplotlib.pyplot as plt
from Server.server_params import *
import socket

LOAD = 1

if not LOAD:
    # https://www.biosemi.com/faq/file_format.htm
    CHANNELS_IN_FILE = 17  # with triggers
    HEADER_LENGTH = 256 * (CHANNELS_IN_FILE + 1)
    SAMPLING_RATE = 2048
    FILE_PATH = "DataBDF/TrainData/Nikodem/Nikodem_0.bdf"

    file_bytes = os.stat(FILE_PATH).st_size
    file_bytes_no_head = file_bytes - HEADER_LENGTH

    channel_sections_count = file_bytes_no_head // (CHANNELS_IN_FILE * SAMPLING_RATE * 3)

    with open(FILE_PATH, 'rb') as f:
        data = f.read()
    data = np.frombuffer(data[HEADER_LENGTH:], dtype='<u1')

    samples = np.ndarray((CHANNELS_IN_FILE - 1, SAMPLING_RATE * channel_sections_count, 3), dtype='<u1')

    for sec in range(channel_sections_count):
        for ch in range(CHANNELS_IN_FILE - 1):
            for sam in range(SAMPLING_RATE):
                beg = sec * CHANNELS_IN_FILE * SAMPLING_RATE * 3 + ch * SAMPLING_RATE * 3 + sam * 3
                samples[ch, sec * SAMPLING_RATE + sam, :] = data[beg:beg + 3]

    np.save("testdata", samples)
else:
    samples = np.load("testdata.npy")

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
