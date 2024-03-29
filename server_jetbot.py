import socket
import struct
import random
import pickle
import Server.server_data_convert as dc
import numpy as np
import torch
import time
import threading
from Server.server_params import *
from Utilities.decision_making import DecisionMaker
from Jetbot.utils.midas_detection import Midas, MidasInterpreter, DecisionMerger
from Jetbot.utils.communication import FrameClient, CommandClient
from Jetbot.utils.configuration import COMMANDS, INVCOMMANDS

JETBOT_ADDRESS = '192.168.0.103'
JETBOT_PORT = 3333

SERVER_ADDRESS = '192.168.0.100'
SERVER_PORT = 22243

COMMAND_SERVING_PORT = 22242

free_boxes = np.array([False, False, False]) # The "resource" that zenazn mentions.
free_boxes_lock = threading.Lock()

def create_sockets():
    # TCP Socket for receiving data from Actiview
    tcp_client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_client_sock.bind(("localhost", TCP_LOCAL_PORT))
    tcp_client_sock.connect((TCP_AV_ADDRESS, TCP_AV_PORT))

    # UDP socket for sending classification results to the client
    udp_server_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_server_sock.bind((SERVER_ADDRESS, COMMAND_SERVING_PORT))

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


def obstacle_detection():
    global free_boxes
    midas = Midas(CAMERA=True)
    midas_iterpreter = MidasInterpreter()
    comm = FrameClient((SERVER_ADDRESS,SERVER_PORT))
    frame = None
    while True:
        frame = comm.receive_frame()
        if frame is None:
            continue
        depth_image = midas.predict(frame,False,0,0)
        if depth_image is None:
            continue
        new_free_boxes = midas_iterpreter.find_obstacles(depth_image)
        if new_free_boxes is None:
            continue
        with free_boxes_lock:
            free_boxes = new_free_boxes



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

    # decision_maker = DecisionMaker(window_length=60, priorities=[2, 0, 1], thresholds=[0.6, 0.50, 0.55])
    decision_maker = DecisionMaker(window_length=60, priorities=[2, 0, 1], thresholds=[0.9, 0.48, 0.4])
    decisions_to_ignore = 0
    decision_ignored = None
    prev_decision = None

    received_data_struct_buffer = bytearray()

    # Start a thread to predict on frames
    predicting = threading.Thread(target=obstacle_detection)
    predicting.start()
    decision_merger = DecisionMerger()

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
            x = x[:, :, 5:, :]
            y = dc.get_classification(x, model)
            out_ind = np.argmax(y.numpy())
            # print(out_ind)
            decision_maker.add_data(out_ind)

            if time.time() - time_start > 0.75:
                decision = str(decision_maker.decide())
                if decisions_to_ignore > 0 and prev_decision != decision and decision_ignored != decision:
                    decisions_to_ignore -= 1
                else:
                    if decision == '0' or decision == '1':
                        decisions_to_ignore = 5
                        decision_ignored = decision
                    print(f"Decision: {kierunki[decision]}")
                    print(decision_maker.decisions_masks)

                    if decision is not None and decision != 'None':
                        with free_boxes_lock:
                            decision = decision_merger.merge(COMMANDS[int(decision)],free_boxes)
                    if decision is not None and decision != 'None':
                        bytes_to_send = str.encode(str(INVCOMMANDS[decision]))
                        udp_server_sock.sendto(bytes_to_send, (JETBOT_ADDRESS, JETBOT_PORT))
                time_start = time.time()
                prev_decision = decision

            seq_num += 1
            if seq_num == 2 ^ 32:
                seq_num = 0


if __name__ == '__main__':
    main()
