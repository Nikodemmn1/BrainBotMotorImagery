# import socket
#
# SERVER_ADDRESS_PORT = ("192.0.0.240", 22221)
# BUFFER_SIZE = 1024
# UDP_SERVER_SOCKET = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
# UDP_SERVER_SOCKET.bind(SERVER_ADDRESS_PORT)
#
#
# def receive_message():
#     message = UDP_SERVER_SOCKET.recvfrom(BUFFER_SIZE)
#     return message[0], message[1]
#
#
# msgFromServer = "Hello UDP Client"
# bytesToSend = str.encode(msgFromServer)
# print("UDP server up and listening")
#
#
# def send_message(message, address):
#     bytes_to_send = str.encode(message)
#     UDP_SERVER_SOCKET.sendto(bytes_to_send, address)
#
#
# while True:
#     message, address = receive_message()
#     print(message, address)
#     UDP_SERVER_SOCKET.sendto(bytesToSend, address)

import socket

# The type of communications between the two endpoints, typically SOCK_STREAM for connection-oriented protocols and SOCK_DGRAM for connectionless protocols.
serversocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
host = "127.0.0.1"
port = 8000
print(host)
print(port)
serversocket.bind((host, port))

# serversocket.listen(5)  #--This method sets up and start TCP listener.
print('server started and listening')
while 1:
    # (clientsocket, address) = serversocket.accept() #---This passively accept TCP client connection, waiting until connection arrives (blocking)
    # print("connection found!")
    # data = clientsocket.recv(1024).decode() #This method receives TCP message
    data, addr = serversocket.recvfrom(2048)
    print(data)
    r = 'I can hear you by UDP!!!!'
    serversocket.sendto(r.encode(), addr)
serversocket.close()
