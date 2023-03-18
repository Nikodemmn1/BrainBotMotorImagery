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

BUFFER_SIZE = 1024
DATAGRAM_MAX_SIZE = 65540

# Connection configuration:
SERVER_ADDRESS = ("192.168.0.163",22241)
JETBOT_ADDRESS = ("192.168.0.145", 3333)


FRAME_SHAPE = (384, 384, 3)


free_boxes = None # The "resource" that zenazn mentions.
free_boxes_lock = threading.Lock()


frame_count = 3
# For the small Midas:
MIN_SAFE_DISTANCE = 7 #500
# For the beit Midas:
#MIN_SAFE_DISTANCE = 6000

COMMANDS = {
    0: 'left',
    1: 'right',
    2: 'forward'
}
INVCOMMANDS = {
    'left': 0 ,
    'right': 1,
    'forward':2
}

listener = None
PLOT = False


KEYS = [False,False,False]

# Debug mode:
DEBUG_PRINT = False


def find_last_img(path='./img/frame'):
    biggest_num = 0
    files = os.listdir(path)
    if len(files) == 0:
        return 0
    for file in files:
        if file.split('.')[-1] == 'png':
            num = int(file.split('.')[-2].split('_')[-1])
            if biggest_num < num:
                biggest_num = num
    return biggest_num + 1
IMG_ITERATOR = find_last_img('./img/frame')


class Midas:
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
        if PLOT:
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
        self.avg /= frame_count
        self.free_boxes = self.avg < MIN_SAFE_DISTANCE
        print(f"Depth prediction:\n"
              f"Distances: left={self.avg[0]} middle={self.avg[1]} right{self.avg[2]}\n"
              f"is_free: left={self.free_boxes[0]} middle={self.free_boxes[1]} right{self.free_boxes[2]}")
        return self.free_boxes

    def reset_values(self):
        self.avg = np.array([0., 0., 0.])
        self.free_boxes = np.array([False, False, False])




class JetsonMock:
    def __init__(self):
        self.przeszkoda = 0

    def move_robot(self, command):
        with free_boxes_lock:
            left  = free_boxes[0]
            front = free_boxes[1]
            right = free_boxes[2]

        if command == 'forward':
            if self.przeszkoda > 0:
                print(f"Przeszkoda czekam {self.przeszkoda}")
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

    def get_command(self):
        # Check for arrow key input
        if KEYS[2]:
            print('Up arrow key pressed')
            return 'forward'
        elif KEYS[0]:
            print('Left arrow key pressed')
            return 'left'
        elif KEYS[1]:
            print('Right arrow key pressed')
            return 'right'
        return None

    def move(self, command):
        command = self.move_robot(command)
        return command


class FrameClient:
    BUFFER_SIZE = 1024
    DATAGRAM_MAX_SIZE = 65540

    def __init__(self):
        self.SERVER_ADDRESS_PORT = SERVER_ADDRESS
        self.UDP_CLIENT_SOCKET = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.UDP_CLIENT_SOCKET.bind(self.SERVER_ADDRESS_PORT)

    def receive_frame(self):
        message = self.UDP_CLIENT_SOCKET.recv(DATAGRAM_MAX_SIZE)
        if len(message) < 100:
            frame_info = pickle.loads(message)
            if frame_info:
                nums_of_packs = frame_info["packs"]
                if DEBUG_PRINT:
                    print(f"New Frame with {nums_of_packs} pack(s)")

                for i in range(nums_of_packs):
                    message = self.UDP_CLIENT_SOCKET.recv(DATAGRAM_MAX_SIZE)
                    if i == 0:
                        buffer = message
                    else:
                        buffer += message

                frame = np.frombuffer(buffer, dtype=np.uint8)
                frame = frame.reshape(frame.shape[0], 1)
                frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                frame = cv2.flip(frame, 1)

                if frame is not None and type(frame) == np.ndarray:
                    if DEBUG_PRINT:
                        print(f"Frame Received")
                    return frame
        return None

    def active_wait(self,seconds=0.2):
        ready = select.select([self.UDP_CLIENT_SOCKET], [], [], 0.2)
        return ready[0]


class CommandClient:
    def __init__(self):
        self.JETBOT_ADDRESS_PORT = JETBOT_ADDRESS
        self.UDP_CLIENT_SOCKET = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

    def send_command(self, command):
        message = str.encode(str(command))
        self.UDP_CLIENT_SOCKET.sendto(message, self.JETBOT_ADDRESS_PORT)


def on_press(key):
    global KEYS
    global PLOT
    try:
        if key == keyboard.Key.up:
            #print('Up arrow key pressed')
            KEYS = [False, False, True]
        elif key == keyboard.Key.down:
            #print('Down arrow key pressed')
            KEYS = [False, False, False]
        elif key == keyboard.Key.left:
            #print('Left arrow key pressed')
            KEYS = [True, False, False]
        elif key == keyboard.Key.right:
            #print('Right arrow key pressed')
            KEYS = [False ,True, False]
        elif key == keyboard.Key.space:
            PLOT = True
    except AttributeError:
        # Ignore keys that don't have an ASCII representation
        pass


def on_release(key):
    global KEYS
    global PLOT
    try:
        if key == keyboard.Key.up:
            KEYS = [False,False,False]
        elif key == keyboard.Key.left:
            KEYS = [False,False,False]
        elif key == keyboard.Key.right:
            KEYS = [False,False,False]
        elif key == keyboard.Key.space:
            PLOT = False

    except AttributeError:
        # Ignore keys that don't have an ASCII representation
        pass

# Create a keyboard listener that runs in the background
def CreateKeyboardListener():
    global listener

    if listener == None:
        listener = keyboard.Listener(on_press=on_press, on_release=on_release,suppress=True)
        listener.start()


def obstacle_detection():
    global free_boxes
    midas = Midas()
    comm = FrameClient()
    local_frame_count = 0
    while True:
        if comm.active_wait(seconds=0.2):
            frame = comm.receive_frame()
            if frame is None:
                continue
            depth_image = midas.predict(frame)
            if depth_image is None:
                continue
            midas.update(depth_image)
            local_frame_count += 1
            if local_frame_count >= frame_count:
                local_frame_count = 0
                new_free_boxes = midas.update_free_boxes()
                if new_free_boxes is None:
                    continue
                with free_boxes_lock:
                    free_boxes = new_free_boxes
                midas.reset_values()
                time.sleep(0.05)




if __name__ == '__main__':
    print("Setting up server...")
    jetson_mock = JetsonMock()
    udp_client = CommandClient()
    CreateKeyboardListener()

    # Start a thread to predict on frames
    predicting = threading.Thread(target=obstacle_detection)
    predicting.start()
    time.sleep(1)
    print("Server configured")

    while True:
        time.sleep(0.5)
        command = jetson_mock.get_command()
        if command is None: continue
        # if DEBUG_PRINT:
        print(f"Capturing command {command}")
        command = jetson_mock.move(command)
        if command is None: continue
        print(f"Sending Command {command}")
        command_index = INVCOMMANDS[command]
        udp_client.send_command(command_index)


