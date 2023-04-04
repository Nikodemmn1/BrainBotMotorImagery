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
SERVER_ADDRESS = ("192.168.0.163", 22243)
JETBOT_ADDRESS = ("192.168.0.145", 3333)
FRAME_COUNT = 0

# Debug mode:
DEBUG_PRINT = False

# Load the jetbot code
import os
import sys
# Hack to get to the jetson python module:
sys.path.append('/home/jetson/Documents/JetbotProject/lib')

# Load jetbot module (code inside jetbot - just a jetson repo code)
from jetbot import Robot


# Inports:

import asyncio

import time

import threading

# Accepted commands:
COMMANDS = {
     0: 'left',
     1: 'right',
     2: 'forward',
}

from robot_communication import FrameClient, CommandClient

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
          


def reporting_camera_frames(server_address=SERVER_ADDRESS):
    global FRAME_COUNT
    conn = FrameClient(server_address)
    i = 0
    frame_count = 3

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


def hearken_orders(jetson,jetbot_address=JETBOT_ADDRESS):
    conn = CommandClient(jetbot_address)
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
    time.sleep(14)
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