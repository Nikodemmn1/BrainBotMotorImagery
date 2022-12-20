import socket
import random

""""
Server send data as string where:
    '0' - left
    '1' - front (relax)
    '2' - right
    
Client should always send request for new data. 
Then server receives request and responds with proper data described above.
"""


class UDPServer:
    def __init__(self, server_address="0.0.0.0", port=22221, buffer_size=1024):
        self.SERVER_ADDRESS_PORT = (server_address, port)
        self.BUFFER_SIZE = buffer_size
        self.UDP_SERVER_SOCKET = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.UDP_SERVER_SOCKET.bind(self.SERVER_ADDRESS_PORT)
        print("UDP server started")

    def receive_message(self):
        message = self.UDP_SERVER_SOCKET.recvfrom(self.BUFFER_SIZE)
        return message[0], message[1]

    def send_message(self, message, address):
        bytes_to_send = str.encode(message)
        self.UDP_SERVER_SOCKET.sendto(bytes_to_send, address)


while True:
    udp_server = UDPServer()
    message_received, address = udp_server.receive_message()
    print(message_received, address)

    message_send = str(random.randint(0, 2))
    udp_server.send_message(message_send, address)
