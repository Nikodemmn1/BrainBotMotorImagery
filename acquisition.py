#!/home/pete/anaconda3/envs/BrainBot/bin/python3.8
import random
import time
import os
from Server.server_params import *
import socket
import pickle
from Calibration.server_params import *
from Calibration.calibration_system_params import *
class bcolors:
    IMAGERY_CLASS = '\033[37m'
    ACC = '\033[32m'
    WARNING = '\033[37m'

IMAGERY_CLASSES = {'0': 'LEFT HAND',
                   '1': 'RIGHT HAND',
                   '2': 'RELAX'}
TRIALS_NUM = 60
TRIALS_PER_CLASS = int(TRIALS_NUM/len(IMAGERY_CLASSES))
TRIAL_DURATION = 5
BREAK_DURATION = 5
PREPARATION_DURATION = 1
TIME_BETWEEN_PACKETS = 0.0625

def acquisition_interface(server_socket, dashboard_socket):
    POOL = []
    for key in IMAGERY_CLASSES.keys():
        POOL.extend([int(key)] * TRIALS_PER_CLASS)
    random.shuffle(POOL)
    t1 = time.time()
    os.system('clear')
    while time.time() - t1 < PREPARATION_DURATION:
        print(bcolors.WARNING + "PREPARE FOR CALIBRATION PROCEDURE..")
        time_left = PREPARATION_DURATION - (time.time() - t1)
        print(bcolors.WARNING + str(int(time_left)))
        time.sleep(0.5)
        os.system('clear')
    for i in range(TRIALS_NUM):
        trial_start = time.time()
        class_id = random.sample(POOL, 1)[0]
        os.system('clear')
        print(bcolors.IMAGERY_CLASS + IMAGERY_CLASSES[str(class_id)])
        print("MODEL ACCURACY" + bcolors.ACC + " {0:0.02f} %".format(random.random()*100))
        while time.time() - trial_start <  TRIAL_DURATION:
            send_class_id(class_id, server_socket, dashboard_socket)
            time.sleep(TIME_BETWEEN_PACKETS)
        os.system('clear')
        print(bcolors.IMAGERY_CLASS + "BREAK")
        print("MODEL ACCURACY" + bcolors.ACC + " {0:0.02f} %".format(random.random() * 100))
        break_start = time.time()
        while time.time() - break_start < BREAK_DURATION:
            send_class_id(None, server_socket, dashboard_socket)
            time.sleep(TIME_BETWEEN_PACKETS)

def create_sockets():
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_sock.bind((UDP_ACQUISITION_IP, UDP_ACQUISITION_PORTS[0]))
    server_sock.connect((UDP_CALIBRATION_SERVER_IP, UDP_CALIBRATION_SERVER_PORTS[0]))

    udp_dashboard_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_dashboard_sock.bind((UDP_ACQUISITION_IP, UDP_ACQUISITION_PORTS[1]))
    udp_dashboard_sock.connect((UDP_DASHBOARD_IP, UDP_DASHBOARD_PORTS[1]))
    return server_sock, udp_dashboard_sock

def send_class_id(class_id, server_socket, dashboard_socket):
    if class_id == None:
        packet = None
    else:
        packet = class_id
    pickled_packet = pickle.dumps(packet)
    server_socket.sendto(pickled_packet, (UDP_CALIBRATION_SERVER_IP, UDP_CALIBRATION_SERVER_PORTS[0]))
    if USE_DASHBOARD:
        dashboard_socket.sendto(pickled_packet, (UDP_DASHBOARD_IP, UDP_DASHBOARD_PORTS[1]))
def main():
    server_socket, dashboard_socket = create_sockets()
    acquisition_interface(server_socket, dashboard_socket)

if __name__ == "__main__":
    main()