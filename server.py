import socket
import struct
import random
import pickle
import Server.server_data_convert as dc
import numpy as np
import torch
from Server.server_params import *


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

    model = load_model()
    mean_std = load_mean_std()

    sec_res = np.zeros(3)
    sec_samp = 0

    while True:
        # Decoding the received packet from ActiView
        received_data_struct = tcp_client_sock.recv(WORDS * 3)
        raw_data = struct.unpack(str(WORDS * 3) + 'B', received_data_struct)
        decoded_data = dc.decode_data_from_bytes(raw_data)
        # decoded_data[CHANNELS-1, :] = np.bitwise_and(decoded_data[CHANNELS-1, :].astype(int), 2 ** 17 - 1)

        buffer = np.roll(buffer, -SAMPLES, axis=1)
        buffer[:, -SAMPLES:] = decoded_data

        if buffer_filled + SAMPLES < SERVER_BUFFER_LEN:
            buffer_filled += SAMPLES
            sec_samp += 1
        else:
            x = dc.prepare_data_for_classification(buffer, mean_std["mean"], mean_std["std"])
            y = dc.get_classification(x, model)
            out_ind = np.argmax(y.numpy())
            sec_res[out_ind] += 1

            sec_samp += 1
            if sec_samp >= 16:
                print(f"Max: {np.argmax(sec_res)+1}: [{sec_res[0]}, {sec_res[1]}, {sec_res[2]}]")
                sec_res = np.zeros(3)
                sec_samp = 0

            # left = True if label == 1 else False
            # forward = True
            # send_string = '{"left": ' + str(left).lower() + ', "forward": ' + str(forward).lower() + "} "
            # message_bytes = send_string.encode()
            # result_to_send = struct.pack("I", seq_num) + message_bytes
            # udp_server_sock.sendto(result_to_send, (REMOTE_UDP_ADDRESS, REMOTE_UDP_PORT))

            seq_num += 1
            if seq_num == 2 ^ 32:
                seq_num = 0


if __name__ == '__main__':
    main()
