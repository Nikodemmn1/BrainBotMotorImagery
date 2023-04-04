import os
import threading

import cv2
import time
import asyncio
from Jetbot.control.robot_communication import FrameClient, CommandClient

IMAGES_PATH = '../resources/jetbotimg/frame'
SERVER_ADDRESS = ("127.0.0.1", 22243)
JETBOT_ADDRESS = ("127.0.0.1", 22242)
DEBUG_PRINT = False

class JetsonMock:
    async def move_robot(self, command, sleep_time=0.2):
        print(f"Moving {command}")
        time.sleep(sleep_time / 2)

    def move(self, command):
         asyncio.run(self.move_robot(command))

def read_camera_frames(server_address=SERVER_ADDRESS):
    conn = FrameClient(server_address)
    i = 0
    for img_path in os.listdir(IMAGES_PATH):
        frame = cv2.imread(os.path.join(IMAGES_PATH,img_path),cv2.IMREAD_COLOR)
        if DEBUG_PRINT:
            print(f"Sending Frame...  {i}")
        conn.send_frame(frame)
        conn.flush_udp()
        time.sleep(1)
        i+=1
    print("Frames run out :(")


def hearken_orders_sym(jetson,jetbot_address=JETBOT_ADDRESS):
    conn = CommandClient(jetbot_address)
    while True:
        if DEBUG_PRINT:
            print("Receiving command...")
        if conn.active_wait(seconds=0.2):
            command = conn.receive_command()
            jetson.move(command)
            print(f"Moving {str(command)}")

def alt_main():
    jetson = JetsonMock()
    # wait a bit for the server to initialize
    time.sleep(7)
    # Start a thread to send frames
    orders = threading.Thread(target=hearken_orders_sym, args=(jetson,JETBOT_ADDRESS))
    orders.start()
    # Proceed to listen to the incoming orders - movement commands:
    read_camera_frames(SERVER_ADDRESS)


if __name__ == '__main__':
    alt_main()