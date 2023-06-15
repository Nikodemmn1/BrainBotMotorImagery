import socket
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import select
import asyncio
import time
import os
from pynput import keyboard
import threading

class Midas:
    PLOT = False
    frame_count = 3
    MIN_SAFE_DISTANCE = 7
    RESOLUTION = (384, 384)
    MEAN_PIXEL_COUNT_RATIO = 0.1
    MEAN_PIXEL_COUNT = int(RESOLUTION[0] * 0.35 * RESOLUTION[1] * 0.2 * MEAN_PIXEL_COUNT_RATIO)
    Y_BOX_POSITION = (int(RESOLUTION[1] * 0.4), int(RESOLUTION[1] * 0.6))
    X_BOX_POSITION = (int(RESOLUTION[0] * 0.05), int(RESOLUTION[0] * 0.35), int(RESOLUTION[0] * 0.7), int(RESOLUTION[0] * 0.95))

    def __init__(self):
        midas, transform, device = self.load_model_midas()
        self.midas = midas
        self.transform = transform
        self.device = device

        self.avg = np.array([0., 0., 0.])
        self.free_boxes = np.array([False, False, False])

    def load_model_midas(self):
        print('load_midas')
        # model_t
        # model_type = "DPT_BEiT_L_512" # MiDaS v3.1 - Large (For highest quality - 3.2023)
        model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        # model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
        # model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        midas.to(device)
        midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            transform = midas_transforms.dpt_transform
        elif model_type == "DPT_BEiT_L_512":
            transform = midas_transforms.beit512_transform
        else:
            transform = midas_transforms.small_transform
        return midas, transform, device

    def predict(self, img):
        global IMG_ITERATOR
        #img = img[30:,:,:]
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
        if self.PLOT:
            plt.figure()
            plt.imshow(img)
            plt.axis('off')
            plt.imsave(f"./img/frame/frame_{IMG_ITERATOR}.png",img)
            plt.show()

            plt.figure()
            plt.imshow(prediction.cpu().numpy())
            plt.axis('off')
            plt.imsave(f"./img/pred/pred_{IMG_ITERATOR}.png",prediction.cpu().numpy())
            plt.show()
            IMG_ITERATOR += 1
        return prediction.cpu().numpy()

    @staticmethod
    def mean_biggest_values(array):
        array = array.flatten()
        ind = np.argpartition(array, - Midas.MEAN_PIXEL_COUNT)[Midas.MEAN_PIXEL_COUNT:]
        return np.average(array[ind])

    def update(self, depth_image):
        self.avg += self.average(depth_image)

    def average(self, depth_image):
        left = self.mean_biggest_values(
            depth_image[self.Y_BOX_POSITION[0]:self.Y_BOX_POSITION[1], self.X_BOX_POSITION[0]:self.X_BOX_POSITION[1]])
        mid = self.mean_biggest_values(
            depth_image[self.Y_BOX_POSITION[0]:self.Y_BOX_POSITION[1], self.X_BOX_POSITION[1]:self.X_BOX_POSITION[2]])
        right = self.mean_biggest_values(
            depth_image[self.Y_BOX_POSITION[0]:self.Y_BOX_POSITION[1], self.X_BOX_POSITION[2]:self.X_BOX_POSITION[3]])
        return np.array([left, mid, right])

    def update_free_boxes(self):
        self.avg /= self.frame_count
        self.free_boxes = self.avg < self.MIN_SAFE_DISTANCE
        print(f"Depth prediction:\n"
              f"Distances: left={self.avg[0]} middle={self.avg[1]} right{self.avg[2]}\n"
              f"is_free: left={self.free_boxes[0]} middle={self.free_boxes[1]} right{self.free_boxes[2]}")
        return self.free_boxes

    def reset_values(self):
        self.avg = np.array([0., 0., 0.])
        self.free_boxes = np.array([False, False, False])

class MergeDecisions:
    def __init__(self):
        self.przeszkoda = 0

    def concatinate_decisionNdetection(self, command, free_boxes):
        left = free_boxes[0]
        front = free_boxes[1]
        right = free_boxes[2]

        if command == 'forward':
            if self.przeszkoda > 0:
                print(f"Przeszkoda czekam framedetection {self.przeszkoda}")
                self.przeszkoda -= 1
                return None
            if front:
                print("Robot jedzie do przodu")
                self.przeszkoda = 0
                return 'forward'
            else:
                print("Przeszkoda akcja nie jest podjęta!! 909090909090909090909090909090909090")
                self.przeszkoda = 3
                return None
        elif command == 'left':
            print("Robot skreca w lewo")
            self.przeszkoda = 0
            return 'left'
        elif command == 'right':
            print("Robot skreca w prawo")
            self.przeszkoda = 0
            return 'right'
        else:
            return None

class FrameClient:
    DEBUG_PRINT = False
    BUFFER_SIZE = 1024
    DATAGRAM_MAX_SIZE = 65540

    def __init__(self,SERVER_ADDRESS, SERVER_PORT):
        self.SERVER_ADDRESS_PORT = (SERVER_ADDRESS, SERVER_PORT)
        self.UDP_CLIENT_SOCKET = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.UDP_CLIENT_SOCKET.bind(self.SERVER_ADDRESS_PORT)

    def receive_frame(self):
        message = self.UDP_CLIENT_SOCKET.recv(self.DATAGRAM_MAX_SIZE)
        if len(message) < 100:
            frame_info = pickle.loads(message)
            if frame_info:
                nums_of_packs = frame_info["packs"]
                if self.DEBUG_PRINT:
                    print(f"New Frame with {nums_of_packs} pack(s)")

                for i in range(nums_of_packs):
                    message = self.UDP_CLIENT_SOCKET.recv(self.DATAGRAM_MAX_SIZE)
                    if i == 0:
                        buffer = message
                    else:
                        buffer += message

                frame = np.frombuffer(buffer, dtype=np.uint8)
                frame = frame.reshape(frame.shape[0], 1)
                frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                frame = cv2.flip(frame, 1)

                if frame is not None and type(frame) == np.ndarray:
                    if self.DEBUG_PRINT:
                        print(f"Frame Received")
                    return frame
        return None

    def active_wait(self,seconds=0.2):
        ready = select.select([self.UDP_CLIENT_SOCKET], [], [], 0.2)
        return ready[0]


