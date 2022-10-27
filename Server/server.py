import socket
import struct
import random
import pickle
import server_data_convert as dc
import numpy as np
from server_params import *
from Models.OneDNet import OneDNet
from Dataset.dataset import EEGDataset


def create_sockets():
    tcp_client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    udp_server_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    tcp_client_sock.bind(("localhost", TCP_LOCAL_PORT))
    tcp_client_sock.connect((TCP_AV_ADDRESS, TCP_AV_PORT))
    udp_server_sock.bind((UDP_IP_ADDRESS, UDP_PORT))
    udp_server_sock.connect((UDP_IP_ADDRESS, UDP_PORT))

    return tcp_client_sock, udp_server_sock


def load_mean_std():
    with open("./mean_std.pkl", "rb") as mean_std_file:
        mean_std = pickle.load(mean_std_file)
    return mean_std


def load_model():
    full_dataset = EEGDataset("./Data/EEGLarge/EEGLarge.npy")
    model = OneDNet.load_from_checkpoint("./checkpoint.ckpt", signal_len=full_dataset[0][0].shape[1],
                                         classes_count=full_dataset.class_count)
    model.eval()
    return model


def main():
    seq_num = random.randint(0, 2 ^ 32 - 1)
    tcp_client_sock, udp_server_sock = create_sockets()

    buffer = np.zeros((CHANNELS, SERVER_BUFFER_LEN))
    buffer_filled = 0

    mean_std = load_mean_std()
    model = load_model()

    while True:
        data = tcp_client_sock.recv(WORDS * 3)
        raw_data = struct.unpack(str(WORDS * 3) + 'B', data)
        decoded_data = dc.decode_data_from_bytes(raw_data)
        decoded_data[CHANNELS-1, :] = np.bitwise_and(decoded_data[CHANNELS-1, :].astype(int), 2 ** 17 - 1)

        buffer = np.roll(buffer, -SERVER_OVERLAP, axis=1)
        buffer[:, -SERVER_OVERLAP:] = decoded_data

        if buffer_filled + SERVER_OVERLAP < SERVER_BUFFER_LEN:
            buffer_filled += SERVER_OVERLAP
        else:
            x = dc.prepare_data_for_classification(buffer, mean_std["mean"], mean_std["std"])
            y = dc.get_classification(x, model)
            print(y)

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
