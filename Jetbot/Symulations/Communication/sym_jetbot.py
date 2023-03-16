#SERVER_ADDRESS = '127.0.0.1'
#SERVER_JETBOT_PORT = 22221
#
#JETBOT_ADDRESS = '192.168.0.163'
#JETBOT_PORT = 3333

from PIL import Image
import numpy as np
import socket
import time




image = np.array(Image.open("girl_Gertruda.png"))
bytes_array = image.flatten().tobytes()


time.sleep(1)
server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

connected = False
while not connected:
    try:
        server_sock.connect(('127.0.0.1', 22249))
        connected = True
    except Exception as e:
        time.sleep(1)

try:
    while True:
        j = 0
        #batch = bytes_array
        for i in range(0,int(262144/32768)):
            batch = bytes_array[32768*i:32768*(i+1)]
            server_sock.send(batch)
            print(f'batch {j} send of size {len(batch)}')
            j += 1
        time.sleep(0.0625)
        print('Image send')
        time.sleep(10)
        message = server_sock.recv(1024).decode("utf-8")
        print(message)


    server_sock.close()
except Exception as e:
    server_sock.close()
    raise e