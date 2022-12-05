import keyboard
import numpy as np
import os
import torch
import sys
from Dataset.dataset import EEGDataset
from Models.OneDNet import OneDNet
import time
import socket
from UDP_Client.communication_parameters import *

def main():
    MODEL_PATH = "model.ckpt"
    freq = 100
    included_classes = [0, 1, 2]
    # included_channels = [3, 6, 7, 8, 11]
    included_channels = range(16)
    full_dataset = EEGDataset("Data/EEGLarge/EEGLarge_train.npy",
                              "Data/EEGLarge/EEGLarge_val.npy",
                              "Data/EEGLarge/EEGLarge_test.npy",
                              included_classes, included_channels)

    train_dataset, val_dataset, test_dataset = full_dataset.get_subsets()
    data = full_dataset.data[test_dataset.indices]
    labels = full_dataset.labels[test_dataset.indices]
    straight_indices = np.where(labels == 0)[0]
    left_indices = np.where(labels == 1)[0]
    right_indices = np.where(labels == 2)[0]

    model = OneDNet.load_from_checkpoint(channel_count=len(included_channels),
                                         included_classes=included_classes,
                                         checkpoint_path=MODEL_PATH)
    model.eval()

    udp_server_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_server_sock.bind((UDP_ADDRESS, UDP_PORT))
    udp_server_sock.connect((UDP_ADDRESS, UDP_PORT))

    while True:
        if keyboard.is_pressed('a'):
            command_idx = np.random.choice(left_indices)
        elif keyboard.is_pressed('d'):
            command_idx = np.random.choice(right_indices)
        else:
            command_idx = np.random.choice(straight_indices)
        command_data = data[command_idx]
        decision = model(torch.from_numpy(command_data[np.newaxis, ...])).detach().numpy()
        decision = np.argmax(decision)
        udp_server_sock.sendto(str(decision.item()).encode('utf-8'), ("localhost", 5003))
        time.sleep(1/freq)
        print(decision)

if __name__ == "__main__":
    main()