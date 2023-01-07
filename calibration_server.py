import socket
import time
import struct
import random
import pickle
import Server.server_data_convert as dc
import numpy as np
import torch
from Server.server_params import *
from Calibration.server_params import *
from Calibration.calibration_system_params import *
import Calibration.calibration_server_data_convert as cd_converter
from Models.OneDNet import OneDNet
import os
CHANNELS_USED = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
CLASSES_INCLUDED = [0, 1, 2]
MODEL_UPDATE_FREQ = 0.1
CHECKPOINTS_PATH = "Calibration/lightning_logs/"
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
        for i in range(1, 4):
            if i in list(np.unique(markers)):
                self.label = i - 1
        if 0 in list(np.unique(markers)):
                self.label = None
def create_sockets():
    # TCP Socket for receiving data from Actiview
    tcp_client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # UDP Socket for communication with acquisition.py
    udp_acquisition_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_dashboard_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    tcp_client_sock.bind(("localhost", TCP_LOCAL_PORT))
    tcp_client_sock.connect((TCP_AV_ADDRESS, TCP_AV_PORT))

    # udp_acquisition_sock.bind((UDP_CALIBRATION_SERVER_IP, UDP_CALIBRATION_SERVER_PORTS[0]))
    # udp_acquisition_sock.connect((UDP_ACQUISITION_IP,  UDP_ACQUISITION_PORTS[0]))

    udp_dashboard_sock.bind((UDP_CALIBRATION_SERVER_IP, UDP_CALIBRATION_SERVER_PORT))
    udp_dashboard_sock.connect((UDP_DASHBOARD_IP, UDP_DASHBOARD_PORT))

    return tcp_client_sock, udp_acquisition_sock, udp_dashboard_sock


def load_mean_std():
    """Loading the file containing the pickled mean and std of the training dataset."""
    with open("mean_std.pkl", "rb") as mean_std_file:
        mean_std = pickle.load(mean_std_file)
    return mean_std


def load_model():
    """Loading the trained OneDNet model"""
    model = torch.load("model.pt")
    model.eval()
    return model


def receive_data_from_acquisition_app(udp_socket):
    data = udp_socket.recv(4096)
    data_decoded = pickle.loads(data)
    return data_decoded

def send_data_to_dashboard(pred, ground_truth, udp_socket):
    pickled_packet = pickle.dumps((pred, ground_truth))
    udp_socket.sendto(pickled_packet, (UDP_DASHBOARD_IP, UDP_DASHBOARD_PORT))

def update_to_checkpoint():
    dirs = os.listdir(CHECKPOINTS_PATH)
    path = CHECKPOINTS_PATH + dirs[-1] + '/checkpoints/last.ckpt'
    model = OneDNet.load_from_checkpoint(channel_count=len(CHANNELS_USED),
                                         included_classes=CLASSES_INCLUDED,
                                         checkpoint_path=path)
    model.eval()
    return model
def main():
    # Sequence number of the last sent UDP packet
    seq_num = random.randint(0, 2 ^ 32 - 1)

    tcp_client_sock, udp_acquisition_sock, udp_dashboard_sock = create_sockets()

    buffer = np.zeros((CHANNELS - 1, SERVER_BUFFER_LEN))
    buffer_filled = 0

    model = load_model()
    model.eval()
    mean_std = load_mean_std()

    sec_res = np.zeros(3)
    sec_samp = 0
    data = np.empty([1, 1, len(CHANNELS_USED), 200]) # shape = (None, 1, , len(CHANNELS_USED) + 1, 200)
    labels = []
    frames_saved = 0
    t1 = 0
    label_holder = LabelHolder()
    model_update_timer = time.time()
    while True:
        # Decoding the received packet from ActiView
        received_data_struct = tcp_client_sock.recv(WORDS * 3)
        raw_data = struct.unpack(str(WORDS * 3) + 'B', received_data_struct)
        decoded_data, triggers = cd_converter.decode_data_from_bytes(raw_data)
        # decoded_data[CHANNELS-1, :] = np.bitwise_and(decoded_data[CHANNELS-1, :].astype(int), 2 ** 17 - 1)
        #class_id = receive_data_from_acquisition_app(udp_acquisition_sock)
        label_holder.update_label(triggers)
        if label_holder.label is not None:
            buffer = np.roll(buffer, -SAMPLES, axis=1)
            buffer[:, -SAMPLES:] = decoded_data

            if buffer_filled + SAMPLES < SERVER_BUFFER_LEN:
                buffer_filled += SAMPLES
                sec_samp += 1
            else:
                x = dc.prepare_data_for_classification(buffer, mean_std["mean"], mean_std["std"])
                x = x[:, :, CHANNELS_USED, -200:] # data frame length with decimation 40 is 200 so we take last 200 samples
                y = dc.get_classification(x, model)
                out_ind = np.argmax(y.numpy())
                sec_res[out_ind] += 1
                if USE_DASHBOARD:
                    send_data_to_dashboard(out_ind, label_holder.label, udp_dashboard_sock)

                if frames_saved == 0:
                    data[0] = x
                else:
                    data = np.append(data, x, axis=0)
                labels.append(label_holder.label)
                frames_saved += 1

                sec_samp += 1
                if sec_samp >= 16:
                    #print(label_holder.labell)
                    print("16 samples collection time:", time.time()-t1)
                    print(f"Max: {np.argmax(sec_res) + 1}: [{sec_res[0]}, {sec_res[1]}, {sec_res[2]}]")
                    print("DATA SHAPE {}".format(data.shape))
                    print("LABELS LEN {}".format(len(labels)))
                    sec_res = np.zeros(3)
                    sec_samp = 0
                    np.save("calibration_data.npy", data, allow_pickle=False, fix_imports=False)
                    labels_to_save = np.array(labels)
                    np.save("calibration_labels.npy", labels_to_save, allow_pickle=True, fix_imports=False)
                    t1 = time.time()
            # if time.time() - model_update_timer > 1/MODEL_UPDATE_FREQ:
            #     model = update_to_checkpoint()


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
