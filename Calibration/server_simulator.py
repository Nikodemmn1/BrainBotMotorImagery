import socket
import pickle
from server_params import *

def create_sockets():

    # UDP socket for receiving data from acquisition.py
    udp_server_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    udp_server_sock.bind((UDP_CALIBRATION_SERVER_IP, UDP_CALIBRATION_SERVER_PORT))
    udp_server_sock.connect((UDP_ACQUISITION_IP, UDP_ACQUISITION_PORT))

    return udp_server_sock

def main():
    udp_socket = create_sockets()
    while True:
        print("Listening on port {}..".format(UDP_CALIBRATION_SERVER_PORT))
        data = udp_socket.recv(4096)
        data_decoded = pickle.loads(data)
        print(data_decoded)

if __name__ == "__main__":
    main()