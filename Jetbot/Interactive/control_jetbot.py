############################################################################
# This script is ment to be run inside jetbot python.
# It purpose is to control the robot by commands send from the server
# It also sends the frames (form the robot's camera) to the server to detect obstacles
# The communication is done via UDP (but the TCP is also implemented)
# This script works in pair with control_server.py script
############################################################################

# First load the robot's camera
# Sometimes it doesn't work - then restart
import nanocamera as nano
RESOLUTION = (384, 384)
camera = nano.Camera(flip=0, width=RESOLUTION[0], height=RESOLUTION[1], fps=10)
frame = camera.read()

print("Camera Loaded")
print(frame.shape)
# 384, 384, 3

# Connection configuration:
SERVER_ADDRESS = ("192.168.0.163", 22241)
JETBOT_ADDRESS = ("192.168.0.145", 3333)

frame_count = 3 #3

# Debug mode:
DEBUG_PRINT = False

print("Connecting Server...")

# Load the jetbot code
import os
import sys
sys.path.insert(0,'..')

from jetbot import Robot


# Inports:
import select
import asyncio
import cv2
import numpy as np
import socket
import time
import pickle
import math

# Accepted commands:
COMMANDS = {
     0: 'left',
     1: 'right',
     2: 'forward',
}




# Unused TCP Client
# may be broken during the refactorisation
class TCPClient:
    FRAME_BYTE_SIZE = 442368
    FRAME_SPLIT = 9
    FRAME_BATCH_SIZE = 49152  # FRAME_BYTE_SIZE /  FRAME_SPLIT must be  < 65535 (max tcp datagram)
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


# Used UDP client
class UDPClient:
    BUFFER_SIZE = 1024
    MAX_LENGTH_FRAME = 65000 # Max datagram is 65540
    def __init__(self):
         print("Connecting Server...")
         self.JETBOT_ADDRESS_PORT = JETBOT_ADDRESS
         self.SERVER_ADDRESS_PORT = SERVER_ADDRESS
         self.UDP_CLIENT_SOCKET = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
         self.UDP_CLIENT_SOCKET.bind(self.JETBOT_ADDRESS_PORT)
         test_message = "I am Jetbot"
         self.UDP_CLIENT_SOCKET.sendto(str.encode(test_message), self.SERVER_ADDRESS_PORT)
         print(f'Test message "{test_message}" - sent!')
         server_test_message = self.UDP_CLIENT_SOCKET.recv(self.BUFFER_SIZE).decode('utf-8')
         print(f'Test message "{server_test_message}" - received!')
         print("Connection Successful")

    def send_message(self, message):
         bytes_to_send = message
         self.UDP_CLIENT_SOCKET.sendto(bytes_to_send, self.SERVER_ADDRESS_PORT)

    def receive_message(self):
         message = self.UDP_CLIENT_SOCKET.recv(self.BUFFER_SIZE)
         return message

    def send_frame(self,frame):
         retval, buffer = cv2.imencode(".jpg", frame)
         if retval:
              # convert to byte array
              buffer = buffer.tobytes()
              # get size of the frame
              buffer_size = len(buffer)

              num_of_packs = 1
              if buffer_size > self.MAX_LENGTH_FRAME:
                    num_of_packs = math.ceil(buffer_size / self.MAX_LENGTH_FRAME)

              frame_info = {"packs": num_of_packs}

              # send the number of packs to be expected
              if DEBUG_PRINT:
                  print("Number of packs:", num_of_packs)
              message = pickle.dumps(frame_info)
              self.send_message(message)

              left = 0
              right = self.MAX_LENGTH_FRAME

              for i in range(num_of_packs):
                    if DEBUG_PRINT:
                        print("left:", left)
                        print("right:", right)

                    # truncate data to send
                    data = buffer[left:right]
                    left = right
                    right += self.MAX_LENGTH_FRAME

                    # send the frames accordingly
                    message = data
                    self.send_message(message)
              if DEBUG_PRINT:
                  print(f"Sent a frame")

         self.send_message(message)

    def receive_command(self):
         message = self.receive_message()
         command_index_str = message.decode("utf-8")
         command_index = int(command_index_str)
         command = COMMANDS[command_index]
         print('RECIEVED COMMAND FROM SERVER', command)
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
        pass

    async def move_robot(self, command, speed=0.3, sleep_time=0.2):
         if command == 'forward':
              robot.right(speed)
              time.sleep(0.017)
              robot.forward(speed)
              time.sleep(sleep_time)
         elif command == 'left':
              robot.left(speed)
              time.sleep(sleep_time / 2)
         elif command == 'right':
              robot.right(speed)
              time.sleep(sleep_time / 2)
         robot.stop()

    def move(self, command):
         asyncio.run(self.move_robot(command))
          

if __name__ == '__main__':
    print('CSI Camera ready? - ', camera.isReady())
    robot =  Robot()
    time.sleep(7) # wait a bit for the server to initialize
    udp_client = UDPClient()
    jetson = Jetson()

    i = 0
    while True:
         i += 1
         frame = camera.read()

         if i == frame_count:
            if DEBUG_PRINT:
                print("Sending Frame...")
            udp_client.send_frame(frame)
            udp_client.flush_udp()  # this is essential
            i = 0

            ready = select.select([udp_client.UDP_CLIENT_SOCKET], [], [], 0.2)
            if ready[0]:
               #if DEBUG_PRINT:
               print("Receiving command...")
               command = udp_client.receive_command()
               print(f"Moving {str(command)}")
               jetson.move(command)
               time.sleep(0.2)

    robot.stop()
    camera.release()
    del camera