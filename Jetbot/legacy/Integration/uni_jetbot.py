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

print(f"Camera Loaded, frames have shape = {frame.shape}")
# 384, 384, 3

# Connection configuration:
SERVER_ADDRESS = ("192.168.0.101", 22242)
JETBOT_ADDRESS = ("192.168.0.103", 3333)
FRAME_COUNT = 0

# Debug mode:
DEBUG_PRINT = False

# Load the jetbot code
import os
import sys
# Hack to get to the python module that is one dir above us:
sys.path.insert(0,'..')
# Load jetbot module (code inside jetbot - just a jetson repo code)
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
import threading

# Accepted commands:
COMMANDS = {
     0: 'left',
     1: 'right',
     2: 'forward',
}


class FrameClient:
    BUFFER_SIZE = 1024
    MAX_LENGTH_FRAME = 65500

    def __init__(self):
        print("Connecting to Server...")
        self.SERVER_ADDRESS_PORT = SERVER_ADDRESS
        self.UDP_CLIENT_SOCKET = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

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
            self.UDP_CLIENT_SOCKET.sendto(message, self.SERVER_ADDRESS_PORT)

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
                self.UDP_CLIENT_SOCKET.sendto(message, self.SERVER_ADDRESS_PORT)
            if DEBUG_PRINT:
                print(f"Sent a frame")

    def flush_udp(self):
         while True:
              x, _, _ = select.select([self.UDP_CLIENT_SOCKET], [], [], 0.001)
              if len(x) == 0:
                    break
              else:
                    self.UDP_CLIENT_SOCKET.recv(self.BUFFER_SIZE)


class CommandClient:
    BUFFER_SIZE = 1024

    def __init__(self):
        self.JETBOT_ADDRESS_PORT = JETBOT_ADDRESS
        self.UDP_CLIENT_SOCKET = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.UDP_CLIENT_SOCKET.bind(self.JETBOT_ADDRESS_PORT)
        print(f"Listening at {self.JETBOT_ADDRESS_PORT[0]}:{self.JETBOT_ADDRESS_PORT[1]}")

    def receive_command(self):
         message = self.UDP_CLIENT_SOCKET.recv(self.BUFFER_SIZE)
         command_index_str = message.decode("utf-8")
         command_index = int(command_index_str)
         command = COMMANDS[command_index]
         print('RECIEVED COMMAND FROM SERVER', command)
         return command

    def active_wait(self,seconds=0.2):
        ready = select.select([self.UDP_CLIENT_SOCKET], [], [], 0.2)
        return ready[0]

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
          


def reporting_camera_frames():
    global FRAME_COUNT
    conn = FrameClient()
    i = 0
    frame_count = 3
    #while True:
    #    i += 1
    #    frame = camera.read()
    #    if i == frame_count:
    #        i = 0
    #        if DEBUG_PRINT:
    #            print("Sending Frame...")
    #        conn.send_frame(frame)
    #        time.sleep(0.2)
    #    else:
    #        time.sleep(0.2)

    while True:
        frame = camera.read()
        i += 1
        if i == frame_count:
            print(f"Sending Frame... {FRAME_COUNT}")
            FRAME_COUNT += 1
            if DEBUG_PRINT:
                print("Sending Frame...")
            conn.send_frame(frame)
            conn.flush_udp()
            i = 0
            #time.sleep(0.2)





def hearken_orders(jetson):
    conn = CommandClient()
    while True:
        if DEBUG_PRINT:
            print("Receiving command...")
        if conn.active_wait(seconds=0.2):
            command = conn.receive_command()
            jetson.move(command)
            print(f"Moving {str(command)}")

def main():
    jetson = Jetson()
    # wait a bit for the server to initialize
    time.sleep(7)
    # Start a thread to send frames
    sending_frames = threading.Thread(target=reporting_camera_frames)
    sending_frames.start()
    # Proceed to listen to the incoming orders - movement commands:
    hearken_orders(jetson)

    # When closing:
    robot.stop()
    camera.release()
    del camera


def alt_main():
    jetson = Jetson()
    # wait a bit for the server to initialize
    time.sleep(7)
    # Start a thread to send frames
    orders = threading.Thread(target=hearken_orders, args=(jetson,))
    orders.start()
    # Proceed to listen to the incoming orders - movement commands:
    reporting_camera_frames()

    # When closing:
    robot.stop()
    camera.release()
    del camera


if __name__ == '__main__':
    # Initialize
    print('CSI Camera ready? - ', camera.isReady())
    robot = Robot()
    alt_main()