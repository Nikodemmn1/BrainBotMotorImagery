import socket
import time


class UDPClient:
    def __init__(self, server_address="192.168.0.195", port=22221, buff_size=1024):
        self.SERVER_ADDRESS_PORT = (server_address, port)
        self.BUFFER_SIZE = buff_size
        self.UDP_CLIENT_SOCKET = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

    def send_message(self, message):
        bytes_to_send = str.encode(message)
        self.UDP_CLIENT_SOCKET.sendto(bytes_to_send, self.SERVER_ADDRESS_PORT)

    def receive_message(self):
        message = self.UDP_CLIENT_SOCKET.recvfrom(self.BUFFER_SIZE)
        return message[0], message[1]


client = UDPClient()

i = 0
while True:
    i += 1
    client.send_message(str(i))
    received_message, address = client.receive_message()
    print(received_message, address)
    time.sleep(1)
