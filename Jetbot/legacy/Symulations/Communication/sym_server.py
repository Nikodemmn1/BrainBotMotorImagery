#JETBOT_ADDRESS = '127.0.0.1'
#JETBOT_PORT = 3333
#
#VIEW_ADDRESS = '192.168.0.163'
#VIEW_PORT = 8188
#
#SERVER_ADDRESS = '127.0.0.1'
#SERVER_VIEW_PORT = 7230
#SERVER_JETBOT_PORT = 22221

import socket
import struct
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# TCP Socket for receiving data from Actiview
#view_client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#view_client_sock.bind((SERVER_ADDRESS, SERVER_VIEW_PORT))

# TCP Socket for receiving data from Jetbot
server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_sock.bind(('127.0.0.1', 22249))
server_sock.listen(1)

jetbot_connection, jetbot_adress = server_sock.accept()
print('Connection accepted')
try:
    while True:
        received_bytes = 0
        #received_array = []
        received_data = bytearray()
        i = 0
        while received_bytes < 262144:
            received_partial = jetbot_connection.recv(32768)
            received_bytes += len(received_partial)
            received_data += received_partial
            #received_array.extend( np.frombuffer(received_partial,dtype=np.dtype(np.uint8)).tolist() )
            print(f'recieved {i} batch of size {len(received_partial)} / {len(received_data)}')
            i += 1

        received_data = np.frombuffer(received_data,dtype=np.dtype(np.uint8),count=262144)
        print('Image recieved')
        image = received_data.reshape(256,256,4)
        print(image)
        plt.figure()
        plt.imshow(image[:,:,:])
        plt.show()

        #
        jetbot_connection.send(str.encode("Success"))


except Exception as e:
    jetbot_connection.close()
    server_sock.close()
    raise e


jetbot_connection.close()
server_sock.close()








# Recieving form the
