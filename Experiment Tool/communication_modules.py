import socket
import Server.server_data_convert as dc
from communication_parameters import words
import struct

def connect_to_TCPserver(address, port):
    tcp_client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_client_sock.bind((address, port))
    tcp_client_sock.connect(("localhost", 8888))
    return tcp_client_sock

def get_remote_data(TCPsocket):
    data = TCPsocket.recv(words * 3)
    rawData = struct.unpack(str(words * 3) + 'B', data)
    decoded_data = dc.decode_data_from_bytes(rawData)  # [channels, samples]
    return decoded_data