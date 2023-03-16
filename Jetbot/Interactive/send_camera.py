import select
import socket
import time

import nanocamera as nano
RESOLUTION = (384, 384)
camera = nano.Camera(flip=0, width=RESOLUTION[0], height=RESOLUTION[1], fps=10)
frame = camera.read()

print("Camera Loaded")
print(frame.shape)
# 384, 384, 3
FRAME_BYTE_SIZE = 442368
FRAME_SPLIT = 9
FRAME_BATCH_SIZE = 49152 # FRAME_BYTE_SIZE /  FRAME_SPLIT must be  < 65535 (max tcp datagram)


SERVER_ADDRESS = ("192.168.0.163", 22241)
JETBOT_ADDRESS = ("192.168.0.145", 3333)


print("Connecting Server...")

#print("Connecting Server...")
#JETBOT_ADDRESS_PORT = JETBOT_ADDRESS
#SERVER_ADDRESS_PORT = SERVER_ADDRESS
#BUFFER_SIZE = 10245
#UDP_CLIENT_SOCKET = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
#UDP_CLIENT_SOCKET.bind(JETBOT_ADDRESS_PORT)
#UDP_CLIENT_SOCKET.sendto(str.encode(str(BUFFER_SIZE)), SERVER_ADDRESS_PORT)
#print('Sended!')
#message = UDP_CLIENT_SOCKET.recv(BUFFER_SIZE).decode("utf-8")
#print(f'Received! {message}')





import asyncio
import torch
import urllib.request
import os
import cv2
import time
import numpy as np
from ..jetbot import Robot
import socket
import time
import pickle
import math

MEAN_PIXEL_COUNT_RATIO = 0.1
MEAN_PIXEL_COUNT = int(RESOLUTION[0] * 0.35 * RESOLUTION[1] * 0.2 * MEAN_PIXEL_COUNT_RATIO)
Y_BOX_POSITION = (int(RESOLUTION[1] * 0.4), int(RESOLUTION[1] * 0.6))
X_BOX_POSITION = (
    int(RESOLUTION[0] * 0.05), int(RESOLUTION[0] * 0.35), int(RESOLUTION[0] * 0.7), int(RESOLUTION[0] * 0.95))
frame_count = 3
min_safe_distance = 500
COMMANDS = {
    0: 'left',
    1: 'right',
    2: 'forward',
    30: 'none'
}

class TCPClient:
   def __init__(self):
       self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
       connected = False
       while not connected:
           try:
               self.server_sock.connect(SERVER_ADDRESS)
               connected = True
           except Exception as e:
               time.sleep(0.1)
       print("Connected")

   def send_frame(self,frame):
       bytes_array = frame.flatten().tobytes()
       print(f"Frame {frame.shape} bytes {len(bytes_array)}")
       for i in range(0, int(FRAME_BYTE_SIZE/FRAME_BATCH_SIZE)):
           self.server_sock.send(bytes_array[FRAME_BATCH_SIZE * i:FRAME_BATCH_SIZE * (i + 1)])
           print(f'batch {i} send')
       print('frame send')

   def receive_command(self):
       command_idx = int(self.server_sock.recv(1024).decode("utf-8"))
       command = COMMANDS[command_idx]
       return command

   def __del__(self):
       self.server_sock.close()





class UDPClient:
   def __init__(self, server_address="192.168.0.163", port=22221, in_port=3333, buff_size=1024):
       print("Connecting Server...")
       self.JETBOT_ADDRESS_PORT = JETBOT_ADDRESS
       self.SERVER_ADDRESS_PORT = SERVER_ADDRESS
       self.BUFFER_SIZE = buff_size
       self.UDP_CLIENT_SOCKET = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
       self.UDP_CLIENT_SOCKET.bind(self.JETBOT_ADDRESS_PORT)
       self.UDP_CLIENT_SOCKET.sendto(str.encode(str(buff_size)), self.SERVER_ADDRESS_PORT)
       print('Sended!')
       self.UDP_CLIENT_SOCKET.recv(self.BUFFER_SIZE)
       print('Received!')

   def send_message(self, message):
       bytes_to_send = message
       self.UDP_CLIENT_SOCKET.sendto(bytes_to_send, self.SERVER_ADDRESS_PORT)

   def receive_message(self):
       message = self.UDP_CLIENT_SOCKET.recv(self.BUFFER_SIZE)
       return message

   def send_frame(self,frame):
       max_length = 65000
       retval, buffer = cv2.imencode(".jpg", frame)
       if retval:
           # convert to byte array
           buffer = buffer.tobytes()
           # get size of the frame
           buffer_size = len(buffer)

           num_of_packs = 1
           if buffer_size > max_length:
               num_of_packs = math.ceil(buffer_size / max_length)

           frame_info = {"packs": num_of_packs}

           # send the number of packs to be expected
           print("Number of packs:", num_of_packs)
           message = pickle.dumps(frame_info)
           self.send_message(message)

           left = 0
           right = max_length

           for i in range(num_of_packs):
               print("left:", left)
               print("right:", right)

               # truncate data to send
               data = buffer[left:right]
               left = right
               right += max_length

               # send the frames accordingly
               message = data
               self.send_message(message)
           print(f"Send frame")

       self.send_message(message)

   def receive_command(self):
       message = self.receive_message()
       command_index_str = message.decode("utf-8")
       command_index = int(command_index_str)
       command = COMMANDS[command_index]
       print('RECEIVED COMMAND FROM SERVER', command)
       return command

   def flush_udp(self):
       while True:
           x, _, _ = select.select([self.UDP_CLIENT_SOCKET], [], [], 0.001)
           if len(x) == 0:
               break
           else:
               self.UDP_CLIENT_SOCKET.recv(self.BUFFER_SIZE)


class Jetson:
   def __init__(self):
       self.avg = np.array([0., 0., 0.])
       self.free_boxes = np.array([False, False, False])

   def update(self, depth_image):
       self.avg += self.average(depth_image)

   async def move_robot(self, command, speed=0.3, sleep_time=0.2):
       left = self.free_boxes[0]
       front = self.free_boxes[1]
       right = self.free_boxes[2]

       if command == 'forward':
           if front:
               robot.right(speed)
               time.sleep(0.017)
               robot.forward(speed)
               time.sleep(sleep_time)
           else:
               print("Przeszkoda akcja nie jest podjęta!!")
       elif command == 'left':
           robot.left(speed)
           time.sleep(sleep_time / 2)
       else:
           robot.right(speed)
           time.sleep(sleep_time / 2)

       robot.stop()

   @staticmethod
   def save_frame(frame):
       cv2.imwrite(f'./images/image_{i}.jpg', frame[120:160])
       return 1

   @staticmethod
   def mean_biggest_values(array):
       array = array.flatten()
       ind = np.argpartition(array, -MEAN_PIXEL_COUNT)[MEAN_PIXEL_COUNT:]
       return np.average(array[ind])

   def average(self, depth_image):
       left = self.mean_biggest_values(
           depth_image[Y_BOX_POSITION[0]:Y_BOX_POSITION[1], X_BOX_POSITION[0]:X_BOX_POSITION[1]])
       mid = self.mean_biggest_values(
           depth_image[Y_BOX_POSITION[0]:Y_BOX_POSITION[1], X_BOX_POSITION[1]:X_BOX_POSITION[2]])
       right = self.mean_biggest_values(
           depth_image[Y_BOX_POSITION[0]:Y_BOX_POSITION[1], X_BOX_POSITION[2]:X_BOX_POSITION[3]])
       return np.array([left, mid, right])

   def update_free_boxes(self):
       self.avg /= frame_count
       self.free_boxes = self.avg < min_safe_distance

       print(f"Distances: left={self.avg[0]} middle={self.avg[1]} right{self.avg[2]}")
       print(f"is_free: left={self.free_boxes[0]} middle={self.free_boxes[1]} right{self.free_boxes[2]}")

       # return is_free

   def move(self, command):
       self.update_free_boxes()
       asyncio.run(self.move_robot(command))
       self.avg = np.array([0., 0., 0.])



if __name__ == '__main__':
   print('CSI Camera ready? - ', camera.isReady())
   robot = Robot()
   time.sleep(1)
   udp_client = UDPClient()
   jetson = Jetson()

   i = 0
   while True:
       i += 1
       frame = camera.read()
       #depth_image = midas.predict(frame)
       #jetson.update(depth_image)

       if i == frame_count:
           #udp_client.flush_udp()
           print("Sending Frame...")
           udp_client.send_frame(frame)
           udp_client.flush_udp()  # this is essential

           ready = select.select([udp_client.UDP_CLIENT_SOCKET], [], [], 0.1)
           if ready[0]:
            print("Receiving command...")
            command = udp_client.receive_command()
            jetson.move(command)
           i = 0

   robot.stop()
   camera.release()
   del camera