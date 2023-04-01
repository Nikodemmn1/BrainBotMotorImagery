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
SERVER_ADDRESS = ("192.168.0.163",22242)
JETBOT_ADDRESS = ("192.168.0.145", 3333)


FRAME_SHAPE = (384, 384, 3)

FRAME_COUNT = 0

free_boxes = np.array([False, False, False]) # The "resource" that zenazn mentions.
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
CAMERA = True


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
    def __init__(self):
        midas, transform, device = self.load_model_midas()
        self.midas = midas
        self.transform = transform
        self.device = device

    def load_model_midas(self):
        print('load_midas')
        # model_t
        model_type = "DPT_BEiT_L_512" # MiDaS v3.1 - Large (For highest quality - 3.2023)
        #model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
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
        global FRAME_COUNT
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = img[:,:,:]
        #img = img[:, :, ::-1]
        input_batch = self.transform(img).to(self.device)
        with torch.no_grad():
            prediction = self.midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        print(f"Frame predicted {FRAME_COUNT}")
        FRAME_COUNT += 1
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
            plt.axhline(10,0,1,color='black')
            plt.axhline(300,0,1,color='black')
            plt.axvline(52, 0, 1, color='black')
            plt.axvline(112, 0, 1, color='black')
            plt.axvline(272, 0, 1, color='black')
            plt.axvline(332, 0, 1, color='black')
            plt.show()
            IMG_ITERATOR += 1
        if CAMERA:
            pred = prediction.cpu().numpy()
            pred = (255 * (pred - pred.min()) / (pred.max() - pred.min())).astype(np.uint8)
            cv2.imshow('WebCam', pred)
            cv2.waitKey(1)
            if cv2.waitKey(25) == ord('q'):
                return prediction.cpu().numpy()
        return prediction.cpu().numpy()


class MidasInterpreter:
    MIN_SAFE_DISTANCE_MIN =  7250  #7250   #20 #7000
    MIN_SAFE_DISTANCE =      7500  #7500  #25 #7500
    MIN_SAFE_DISTANCE_MEAN = 7400  # 7000 # 21  #6000
    RESOLUTION = (384, 384)
    GROUP_SIZE = 25 #10
    MEAN_PIXEL_COUNT_RATIO = 0.1
    MEAN_PIXEL_COUNT = int(RESOLUTION[0] * 0.35 * RESOLUTION[1] * 0.2 * MEAN_PIXEL_COUNT_RATIO)
    Y_BOX_POSITION = (10, 300) #230)#330) # split into 10 - 320 - 54
    X_BOX_POSITION = (52, 112, 272, 332) # split into 22 - 90 - 160 - 90 - 22

    def __init__(self):
        self.free_boxes = np.array([False, False, False])

    def find_obstacles(self,depth_image):
        self.free_boxes = np.array([False, False, False])
        left_part = depth_image[self.Y_BOX_POSITION[0]:self.Y_BOX_POSITION[1], self.X_BOX_POSITION[0]:self.X_BOX_POSITION[1]]
        mid_part = depth_image[self.Y_BOX_POSITION[0]:self.Y_BOX_POSITION[1], self.X_BOX_POSITION[1]:self.X_BOX_POSITION[2]]
        right_part = depth_image[self.Y_BOX_POSITION[0]:self.Y_BOX_POSITION[1], self.X_BOX_POSITION[2]:self.X_BOX_POSITION[3]]

        left_depth, left_count = self.look_for_grouping(left_part)
        mid_depth, mid_count = self.look_for_grouping(mid_part)
        right_depth, right_count = self.look_for_grouping(right_part)

        left_average =  self.mean_biggest_values(left_part)
        mid_average =  self.mean_biggest_values(mid_part)
        right_average =  self.mean_biggest_values(right_part)

        left_free = left_depth < self.MIN_SAFE_DISTANCE and left_count < 10 and left_average < self.MIN_SAFE_DISTANCE_MEAN
        mid_free = mid_depth < self.MIN_SAFE_DISTANCE and mid_count < 10 and mid_average < self.MIN_SAFE_DISTANCE_MEAN
        right_free = right_depth < self.MIN_SAFE_DISTANCE and right_count < 10 and right_average < self.MIN_SAFE_DISTANCE_MEAN

        print(f"Depth prediction:\n"
              f"Mean Distances: left={left_average} middle={mid_average} right{right_average}\n"
              f"Max Distances: left={left_depth} middle={mid_depth} right{right_depth}\n"
              f"Distances Count: left={left_count} middle={mid_count} right{right_count}\n"
              f"is_free: left={left_free} middle={mid_free} right{right_free}")

        self.free_boxes = np.array([left_free, mid_free, right_free])
        return self.free_boxes


    @staticmethod
    def look_for_grouping(array):
        best_mean = 0.0
        count = 0
        for x in range(0,array.shape[0],MidasInterpreter.GROUP_SIZE):
            for y in range(0,array.shape[1],MidasInterpreter.GROUP_SIZE):
                grid = array[x:x+MidasInterpreter.GROUP_SIZE, y:y+MidasInterpreter.GROUP_SIZE]
                mean = grid.mean()
                if mean > MidasInterpreter.MIN_SAFE_DISTANCE_MIN:
                    count += 1
                if mean > best_mean:
                    best_mean = mean
        return best_mean, count

    @staticmethod
    def mean_biggest_values(array):
        pixel_count = int(array.shape[0]* array.shape[1] * 0.1)
        array = array.flatten()
        ind = np.argpartition(array, - pixel_count)[pixel_count:]
        return np.average(array[ind])





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
            if front and left and right:
                print("Robot jedzie do przodu")
                self.przeszkoda = 0
                return 'forward'
            elif left and not front and not right:
                return 'left'
            elif right and not front and not left:
                return 'right'
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
                #frame = cv2.flip(frame, 1)

                if frame is not None and type(frame) == np.ndarray:
                    #if CAMERA:
                    #    cv2.imshow('WebCam', frame)
                    #    cv2.waitKey(1)
                    #    if cv2.waitKey(25) == ord('q'):
                    #        return frame
                    if DEBUG_PRINT:
                        print(f"Frame Received ")
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
    midas_iterpreter = MidasInterpreter()
    comm = FrameClient()
    local_frame_count = 0
    frame = None
    while True:
        frame = comm.receive_frame()
        if frame is None:
            continue
        depth_image = midas.predict(frame)
        if depth_image is None:
            continue
        new_free_boxes = midas_iterpreter.find_obstacles(depth_image)
        if new_free_boxes is None:
            continue
        with free_boxes_lock:
            free_boxes = new_free_boxes

def command_control():
    jetson_mock = JetsonMock()
    udp_client = CommandClient()
    while True:
        time.sleep(0.7)
        command = jetson_mock.get_command()
        if command is None: continue
        # if DEBUG_PRINT:
        print(f"Capturing command {command}")
        command = jetson_mock.move(command)
        if command is None: continue
        print(f"Sending Command {command}")
        command_index = INVCOMMANDS[command]
        udp_client.send_command(command_index)



def alt_main():
    print("Setting up server...")
    CreateKeyboardListener()
    # Start a thread to predict on frames
    predicting = threading.Thread(target=command_control)
    predicting.start()
    time.sleep(1)
    print("Server configured")
    obstacle_detection()



def main():
    print("Setting up server...")
    CreateKeyboardListener()
    # Start a thread to predict on frames
    predicting = threading.Thread(target=obstacle_detection)
    predicting.start()
    time.sleep(1)
    print("Server configured")
    command_control()

if __name__ == '__main__':
    main()


