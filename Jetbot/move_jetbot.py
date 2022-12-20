import nanocamera as nano

# To musi być na początku, bo inaczej wywali błąd
RESOLUTION = (384, 384)
camera = nano.Camera(flip=0, width=RESOLUTION[0], height=RESOLUTION[1], fps=10)
frame = camera.read()
print('Pierwszy frame', frame)

import asyncio
import torch
import urllib.request
import os
import cv2
import time
import numpy as np
from jetbot import Robot
import socket
import time

MEAN_PIXEL_COUNT_RATIO = 0.1
MEAN_PIXEL_COUNT = int(RESOLUTION[0] * 0.35 * RESOLUTION[1] * 0.2 * MEAN_PIXEL_COUNT_RATIO)
Y_BOX_POSITION = (int(RESOLUTION[1] * 0.4), int(RESOLUTION[1] * 0.6))
X_BOX_POSITION = (
    int(RESOLUTION[0] * 0.05), int(RESOLUTION[0] * 0.35), int(RESOLUTION[0] * 0.7), int(RESOLUTION[0] * 0.95))
frame_count = 3
min_safe_distance = 500
COMMANDS = {
    0: 'left',
    1: 'forward',
    2: 'right'
}


class UDPClient:
    def __init__(self, server_address="192.168.0.195", port=22221, buff_size=1024):
        self.SERVER_ADDRESS_PORT = (server_address, port)
        self.BUFFER_SIZE = buff_size
        self.UDP_CLIENT_SOCKET = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.send_message('gimme command')

    def send_message(self, message):
        bytes_to_send = str.encode(message)
        self.UDP_CLIENT_SOCKET.sendto(bytes_to_send, self.SERVER_ADDRESS_PORT)

    def receive_message(self):
        message = self.UDP_CLIENT_SOCKET.recvfrom(self.BUFFER_SIZE)
        return message[0], message[1]

    def receive_command(self):
        message = self.receive_message()
        command_index_str = message[0].decode("utf-8")
        command_index = int(command_index_str)
        command = COMMANDS[command_index]
        print('RECEIVED COMMAND FROM SERVER', command)
        return command


class Midas:
    def __init__(self):
        midas, transform, device = self.load_model_midas()
        self.midas = midas
        self.transform = transform
        self.device = device

    def load_model_midas(self):
        print('load_midas')
        # model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        # model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
        model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        midas.to(device)
        midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform
        return midas, transform, device

    def predict(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)
        with torch.no_grad():
            prediction = self.midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        return prediction.cpu().numpy()


class Jetson:
    def __init__(self):
        self.avg = np.array([0., 0., 0.])
        self.free_boxes = np.array([False, False, False])

    def update(self, depth_image):
        self.avg += self.average(depth_image)

    async def move_robot(self, command, speed=0.3, sleep_time=0.2):
        if self.free_boxes[1] and self.free_boxes[0] and self.free_boxes[2]:
            print('FORWARD')
            print('speed', speed)
            robot.forward(speed)
        elif self.free_boxes[1] and self.free_boxes[0]:
            print('LEFT')
            robot.left(speed)
            time.sleep(0.1)
            robot.forward(speed)
        elif self.free_boxes[1] and self.free_boxes[2]:
            print('RIGHT')
            robot.left(speed)
            time.sleep(0.1)
            robot.forward(speed)
        elif self.free_boxes[0]:
            print('BACK LEFT')
            robot.left(speed)
            time.sleep(sleep_time)
            robot.forward(speed)
        elif self.free_boxes[2]:
            print('BACK RIGHT')
            robot.right(speed)
            time.sleep(sleep_time)
            robot.forward(speed)
        else:
            print('BACK AROUND')
            robot.backward(speed)
            time.sleep(0.3)
            robot.left(speed)

        time.sleep(sleep_time)
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
    midas = Midas()
    udp_client = UDPClient()
    jetson = Jetson()

    i = 0
    while True:
        i += 1
        frame = camera.read()
        depth_image = midas.predict(frame)
        jetson.update(depth_image)

        if i == frame_count:
            command = udp_client.receive_command()
            jetson.move(command)
            udp_client.send_message('gimme next command')
            i = 0

    robot.stop()
    camera.release()
    del camera
