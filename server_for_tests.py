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
import Calibration.calibration_server_data_convert as cd_converter


JETBOT_ADDRESS = '192.168.0.101'
JETBOT_PORT = 3333


class LabelHolder():
    def __init__(self):
        self.label = None
    def update_label(self, triggers):
        markers = self.find_markers_in_triggers(triggers)
        return markers
    def find_markers_in_triggers(self, triggers):
        markers = triggers[:, 1]
        markers = np.where(markers > 0,
                           np.log2(np.bitwise_and(markers, -markers)).astype('int8') + 1,
                           np.zeros(markers.shape).astype('int8'))
        markers -= 1
        #print(np.unique(markers))
        for i in range(1, 4):
            if i in list(np.unique(markers)):
                self.label = i - 1
        if 0 in list(np.unique(markers)):
            self.label = None


def create_sockets():
    # TCP Socket for receiving data from Actiview
    tcp_client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # UDP socket for sending classification results to the client
    udp_server_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #udp_server_sock.bind(('192.168.0.100', 22221))
    udp_server_sock.bind(('localhost', 22221))

    tcp_client_sock.bind(("localhost", TCP_LOCAL_PORT))
    tcp_client_sock.connect((TCP_AV_ADDRESS, TCP_AV_PORT))
    #udp_server_sock.bind((UDP_IP_ADDRESS, UDP_PORT))
    #udp_server_sock.connect((JETBOT_ADDRESS, JETBOT_PORT))

    return tcp_client_sock, udp_server_sock


def load_mean_std():
    """Loading the file containing the pickled mean and std of the training dataset."""
    with open("./mean_std.pkl", "rb") as mean_std_file:
        mean_std = pickle.load(mean_std_file)
    return mean_std


def load_model():
    """Loading the trained OneDNet model"""
    model = torch.load("model.pt")
    model.eval()
    return model

kierunki = {'0' : 'lewo', '1': 'prawo', '2':'prosto', 'None':'None'}

def main():
    # Sequence number of the last sent UDP packet
    seq_num = random.randint(0, 2 ^ 32 - 1)

    tcp_client_sock, udp_server_sock = create_sockets()

    buffer = np.zeros((CHANNELS-1, SERVER_BUFFER_LEN))
    buffer_filled = 0

    buffer_mean_dc = np.zeros((CHANNELS-1, MEAN_PERIOD_LEN))

    model = load_model()
    mean_std = load_mean_std()

    sec_res = np.zeros(3)
    sec_samp = 0
    time_start = time.time()
    time_total_start = time.time()

    decision_maker = DecisionMaker(window_length=8, priorities=[2, 0, 1], thresholds=[0.9, 0.9, 0.9])
    decisions_to_ignore = 0
    decision_ignored = None
    prev_decision = None

    received_data_struct_buffer = bytearray()
    label_holder = LabelHolder()

    prec_ok = 0
    prec_not_ok = 0
    precision = float("nan")
    true_decisions = []

    while True:
        # Decoding the received packet from ActiView
        received_bytes = 0
        while received_bytes < WORDS * 3:
            received_data_struct_partial = tcp_client_sock.recv(WORDS * 3)
            received_bytes += len(received_data_struct_partial)
            received_data_struct_buffer += received_data_struct_partial
        received_data_struct = bytes(received_data_struct_buffer[:WORDS*3])
        received_data_struct_buffer = received_data_struct_buffer[WORDS*3:]
        raw_data = struct.unpack(str(WORDS * 3) + 'B', received_data_struct)
        decoded_data, triggers = cd_converter.decode_data_from_bytes(raw_data)
        # decoded_data[CHANNELS-1, :] = np.bitwise_and(decoded_data[CHANNELS-1, :].astype(int), 2 ** 17 - 1)
        label_holder.update_label(triggers)

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
            x = x[:, :, 5:, :]
            y = dc.get_classification(x, model)
            out_ind = np.argmax(y.numpy())
            # print(out_ind)
            decision_maker.add_data(out_ind)

            if time.time() - time_start > 0.75:
                decision = decision_maker.decide()
                if decision is not None and time.time() - time_total_start > 10:
                    if decision == label_holder.label or decision in true_decisions[-3:]:
                        prec_ok += 1
                    else:
                        prec_not_ok += 1
                if prec_ok+prec_not_ok > 0:
                    precision = prec_ok/(prec_ok+prec_not_ok)
                print(f"Time: {time.time() - time_total_start} | Precision: {precision} | True Label: {kierunki[str(label_holder.label)]} | Decision: {kierunki[str(decision)]}")
                print(decision_maker.decisions_masks)
                #if decisions_to_ignore > 0 and prev_decision != decision and decision_ignored != decision:
                #    decisions_to_ignore -= 1
                #else:
                #    if decision == '0' or decision == '1':
                #        decisions_to_ignore = 0
                #        decision_ignored = decision
                #    print(f"Decision: {kierunki[decision]}")
                #    print(decision_maker.decisions_masks)
                #    if decision != 'None':
                #        bytes_to_send = str.encode(decision)
                #        udp_server_sock.sendto(bytes_to_send, (JETBOT_ADDRESS, JETBOT_PORT))
                time_start = time.time()
                true_decisions.append(label_holder.label)

            seq_num += 1
            if seq_num == 2 ^ 32:
                seq_num = 0


if __name__ == '__main__':
    main()
