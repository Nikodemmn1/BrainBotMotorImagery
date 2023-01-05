import socket
import struct
import random
import pickle
import Server.server_data_convert as dc
import numpy as np
import torch
import time
from Server.server_params import *
from Utilities.decision_making import DecisionMaker


def create_sockets():
    # TCP Socket for receiving data from Actiview
    tcp_client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # UDP socket for sending classification results to the client
    udp_server_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    tcp_client_sock.bind(("localhost", TCP_LOCAL_PORT))
    tcp_client_sock.connect((TCP_AV_ADDRESS, TCP_AV_PORT))
    udp_server_sock.bind((UDP_IP_ADDRESS, UDP_PORT))
    udp_server_sock.connect((UDP_IP_ADDRESS, UDP_PORT))

    return tcp_client_sock, udp_server_sock


def load_mean_std():
    """Loading the file containing the pickled mean and std of the training dataset."""
    with open("./mean_std.pkl", "rb") as mean_std_file:
        mean_std = pickle.load(mean_std_file)
    return mean_std


def load_model():
    """Loading the trained OneDNet model"""
    model = torch.load("./model.pt")
    model.eval()
    return model


def main():
    # Sequence number of the last sent UDP packet
    seq_num = random.randint(0, 2 ^ 32 - 1)

    tcp_client_sock, udp_server_sock = create_sockets()

    buffer = np.zeros((CHANNELS, SERVER_BUFFER_LEN))
    buffer_filled = 0

    buffer_mean_dc = np.zeros((CHANNELS, MEAN_PERIOD_LEN))

    model = load_model()
    mean_std = load_mean_std()

    sec_res = np.zeros(3)
    sec_samp = 0
    time_start = time.time()

    JETBOT_ADDRESS = '192.168.250.149'
    JETBOT_PORT = 3333

    decision_maker = DecisionMaker(window_length=60, priorities=[2, 0, 1], thresholds=[0.4, 0.4, 0.8])

    while True:
        # Decoding the received packet from ActiView
        received_data_struct = tcp_client_sock.recv(WORDS * 3)
        raw_data = struct.unpack(str(WORDS * 3) + 'B', received_data_struct)
        decoded_data = dc.decode_data_from_bytes(raw_data)
        # decoded_data[CHANNELS-1, :] = np.bitwise_and(decoded_data[CHANNELS-1, :].astype(int), 2 ** 17 - 1)

        buffer = np.roll(buffer, -SAMPLES, axis=1)
        buffer[:, -SAMPLES:] = decoded_data

        buffer_mean_dc = np.roll(buffer_mean_dc, -SAMPLES, axis=1)
        buffer_mean_dc[:, -SAMPLES:] = decoded_data

        if buffer_filled + SAMPLES < SERVER_BUFFER_LEN:
            buffer_filled += SAMPLES
            sec_samp += 1
        else:
            dc_means = buffer_mean_dc.mean(axis=1)
            buffer_no_dc = dc.remove_dc_offset(buffer, dc_means)
            x = dc.prepare_data_for_classification(buffer_no_dc, mean_std["mean"], mean_std["std"])
            x = x[:, :, 2:, :]
            y = dc.get_classification(x, model)
            out_ind = np.argmax(y.numpy())
            decision_maker.add_data(out_ind)

            if time.time() - time_start > 2.0:
                decision = decision_maker.decide()
                bytes_to_send = str.encode(decision)
                udp_server_sock.sendto(bytes_to_send, (JETBOT_ADDRESS, JETBOT_PORT))
                time_start = time.time()

            seq_num += 1
            if seq_num == 2 ^ 32:
                seq_num = 0


if __name__ == '__main__':
    main()
