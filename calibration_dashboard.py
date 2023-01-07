import numpy as np
import socket
from Calibration.server_params import *
import time
import pickle
from select import select

DISPLAY_FREQ = 200

class bcolors:
    NORMAL = '\033[37m'
    NOT_VALID = '\033[91m'
    VALID = '\033[92m'

labels_dict = {'0': 'LEFT',
               '1': 'RIGHT',
               '2': 'RELAX'}

def create_sockets():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((UDP_DASHBOARD_IP, UDP_DASHBOARD_PORT))
    server_socket.connect((UDP_CALIBRATION_SERVER_IP, UDP_CALIBRATION_SERVER_PORT))

    return server_socket

def display_data(ground_truth, prediction):
    if ground_truth == prediction:
        color = bcolors.VALID
    else:
        color = bcolors.NOT_VALID
    if ground_truth != None:
        print(bcolors.NORMAL + labels_dict[str(ground_truth)] + '---' + color + labels_dict[str(prediction)])
    else:
        print(bcolors.NORMAL + "BREAK")

def receive_from_udp(udp_socket):
    data = udp_socket.recv(4096)
    data_decoded = pickle.loads(data)
    return data_decoded

def main():
    server_socket= create_sockets()
    t1 = time.time()
    while True:
        #select(input, [], [])
        data = server_socket.recv(4096)
        #ground_truth = acquisition_socket.recv(4096)
        prediction, ground_truth = pickle.loads(data)
        if time.time() - t1 > (1/DISPLAY_FREQ):
            display_data(ground_truth, prediction)
            t1 = time.time()
if __name__ == "__main__":
    main()