import socket
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import select
import asyncio
import time
from pynput import keyboard
BUFFER_SIZE = 1024
DATAGRAM_MAX_SIZE = 65540

SERVER_ADDRESS = ("192.168.0.163",22241)
FRAME_BYTE_SIZE = 442368
FRAME_SPLIT = 9
FRAME_BATCH_SIZE = 49152 # FRAME_BYTE_SIZE /  FRAME_SPLIT must be  < 65535 (max tcp datagram)
FRAME_SHAPE = (384, 384, 3)

print("Connecting Jetbot...")
JETBOT_ADDRESS = ("192.168.0.145", 3333)


RESOLUTION = (384, 384)
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
    2: 'forward'
}
listener = None
PLOT = False


#print("Connecting Jetbot...")
#SERVER_ADDRESS_PORT = SERVER_ADDRESS
#JETBOT_ADDRESS_PORT = JETBOT_ADDRESS
#BUFFER_SIZE = 10246
#UDP_CLIENT_SOCKET = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
#UDP_CLIENT_SOCKET.bind(SERVER_ADDRESS_PORT)
#message = UDP_CLIENT_SOCKET.recv(BUFFER_SIZE).decode("utf-8")
#print(f'Received! {message}')
#UDP_CLIENT_SOCKET.sendto(str.encode(str(BUFFER_SIZE)), JETBOT_ADDRESS_PORT)
#print('Sended!')






KEYS = [False,False,False]

class TCPClient:
    def __init__(self):
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.bind(SERVER_ADDRESS)
        self.server_sock.listen(1)
        self.connection_socket, self.connection_address = self.server_sock.accept()

    def send_command(self,command):
        self.connection_socket.send(str.encode(str(command)))

    def receive_frame(self):
        received_bytes = 0
        received_data = bytearray()
        i = 0
        while received_bytes < FRAME_BYTE_SIZE:
            received_partial = self.connection_socket.recv(FRAME_BATCH_SIZE)
            received_bytes += len(received_partial)
            received_data += received_partial
            print(f'recieved batch {i} of size {len(received_partial)} / {len(received_data)}')
            i += 1

        received_array = np.frombuffer(received_data, dtype=np.dtype(np.uint8), count=FRAME_BYTE_SIZE)

        print('Frame recieved')

        frame = np.array(received_array).reshape(FRAME_SHAPE)
        # to see the frames uncomment:
        plt.figure()
        plt.imshow(frame)
        plt.show()
        return frame

    def __del__(self):
        self.connection_socket.close()
        self.server_sock.close()


class Midas:
    def __init__(self):
        midas, transform, device = self.load_model_midas()
        self.midas = midas
        self.transform = transform
        self.device = device

    def load_model_midas(self):
        print('load_midas')
        # model_type = "DPT_BEiT_L_512" # MiDaS v3.1 - Large (For highest quality - 3.2023)
        #model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
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
        #plt.figure()
        #plt.imshow(img)
        #plt.show()
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
            plt.imshow(prediction.cpu().numpy())
            plt.show()
        return prediction.cpu().numpy()

class JetsonMock:
    def __init__(self):
        self.avg = np.array([0., 0., 0.])
        self.free_boxes = np.array([False, False, False])
        self.przeszkoda = 0

    def update(self, depth_image):
        self.avg += self.average(depth_image)

    def move_robot(self, command, speed=0.3, sleep_time=0.2):

        left = self.free_boxes[0]
        front = self.free_boxes[1]
        right = self.free_boxes[2]

        if command == 2 : #'forward':
            #if self.przeszkoda > 0:
            #    print(f"Przeszkoda czekam {self.przeszkoda}")
            #    self.przeszkoda -= 1
            #    return 30
            if front:
                print("Robot jedzie do przodu")
                #robot.right(speed)
                #time.sleep(0.017)
                #robot.forward(speed)
                #time.sleep(sleep_time)
                self.przeszkoda = 0
                return 2
            else:
                print("Przeszkoda akcja nie jest podjęta!! 909090909090909090909090909090909090")
                self.przeszkoda = 3
                return 30
        elif command == 0: #'left':
            #robot.left(speed)
            #time.sleep(sleep_time / 2)
            print("Robot skreca w lewo")
            self.przeszkoda = 0
            return 0
        elif command == 1:
            #robot.right(speed)
            #time.sleep(sleep_time / 2)
            print("Robot skreca w prawo")
            self.przeszkoda = 0
            return 1
        else:
            return 30

        #robot.stop()

        # if command == 'forward':
        #     if left and front and right:
        #         print(f"Executing command: {command} - Going FRONT")
        #         robot.forward(speed)
        #     elif left and front:
        #         print(f"Executing command: {command} - Going FRONT-SLIGHTLY-LEFT")
        #         robot.left(speed)
        #         time.sleep(0.1)
        #         robot.forward(speed)
        #     elif front and right:
        #         print(f"Executing command: {command} - Going FRONT-SLIGHTLY-RIGHT")
        #         robot.right(speed)
        #         time.sleep(0.1)
        #         robot.forward(speed)
        #     elif left:
        #         print(f"Executing command: {command} - Going LEFT")
        #         robot.left(speed)
        #         time.sleep(sleep_time)
        #         robot.forward(speed)
        #     elif right:
        #         print(f"Executing command: {command} - Going RIGHT")
        #         robot.left(speed)
        #         time.sleep(sleep_time)
        #         robot.forward(speed)
        #     else:
        #         print(f"Executing command: {command} - Going TURN-AROUND")
        #         robot.backward(speed)
        #         time.sleep(0.3)
        #         robot.left(speed)
        # elif command == 'left':
        #     if left:
        #         print(f"Executing command: {command} - Going LEFT")
        #         robot.left(speed)
        #     else:
        #         print(f"Executing command: {command} - Going TURN-AROUND")
        #         robot.backward(speed)
        #         time.sleep(0.3)
        #         robot.left(speed)
        # elif command == 'right':
        #     if left:
        #         print(f"Executing command: {command} - Going RIGHT")
        #         robot.right(speed)
        #     else:
        #         print(f"Executing command: {command} - Going TURN-AROUND")
        #         robot.backward(speed)
        #         time.sleep(0.3)
        #         robot.left(speed)

    #@staticmethod
    #def save_frame(frame):
    #    cv2.imwrite(f'./images/image_{i}.jpg', frame[120:160])
    #    return 1

    def czy_przeszkoda(self, offset=0):
       if self.przeszkoda > offset:
           return True
       else:
           return False

    def get_command(self):
        # Check for arrow key input
        if KEYS[2]:
            print('Up arrow key pressed')
            return 2
        elif KEYS[0]:
            print('Left arrow key pressed')
            return 0
        elif KEYS[1]:
            print('Right arrow key pressed')
            return 1
        return None

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
        #asyncio.run(self.move_robot(command))
        command = self.move_robot(command)
        self.avg = np.array([0., 0., 0.])
        return command


class UDPClient:
    def __init__(self, server_address="192.168.0.145", port=3333, in_port=22221, buff_size=1024):
        print("Connecting Jetbot...")
        self.SERVER_ADDRESS_PORT = SERVER_ADDRESS
        self.JETBOT_ADDRESS_PORT = JETBOT_ADDRESS
        self.BUFFER_SIZE = buff_size
        self.UDP_CLIENT_SOCKET = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.UDP_CLIENT_SOCKET.bind(self.SERVER_ADDRESS_PORT)
        self.UDP_CLIENT_SOCKET.recv(self.BUFFER_SIZE)
        print('Received!')
        self.UDP_CLIENT_SOCKET.sendto(str.encode(str(self.BUFFER_SIZE)), self.JETBOT_ADDRESS_PORT)
        print('Sended!')

    def send_message(self, message):
        bytes_to_send = str.encode(str(message))
        self.UDP_CLIENT_SOCKET.sendto(bytes_to_send, self.JETBOT_ADDRESS_PORT)

    def receive_message(self,batch_size):
        message = self.UDP_CLIENT_SOCKET.recv(batch_size)
        return message

    def receive_frame(self):
        #received_bytes = 0
        #received_array = []
        #order_array = []
        #received_data = bytearray()
#
        #received_bytes = 0
        #received_data = bytearray()
        #i = 0
        #while received_bytes < FRAME_BYTE_SIZE:
        #    message = self.receive_message(FRAME_BATCH_SIZE+1)
        #    order = np.frombuffer(message, dtype=np.dtype(np.uint8), count = 1)
        #    partial = np.frombuffer(message, dtype=np.dtype(np.uint8), count = FRAME_BATCH_SIZE, offset = 1)
        #    received_bytes += len(partial)
        #    received_array.append(partial)
        #    order_array.append(order)
        #    print(f'recieved batch {i} of size {len(message)-1} / {received_bytes}')
        #    i += 1
#
        #data_ordered_array = received_array
        #for order, receive  in zip(order_array, received_array):
        #    data_ordered_array[order] = receive
#
        #ordered_data = bytearray()
        #for data in data_ordered_array:
        #    ordered_data += data
        message = self.receive_message(DATAGRAM_MAX_SIZE)
        if len(message) < 100:
            frame_info = pickle.loads(message)

            if frame_info:
                nums_of_packs = frame_info["packs"]
                #print(f"New Frame with {nums_of_packs} pack(s)")

                for i in range(nums_of_packs):
                    message = self.receive_message(DATAGRAM_MAX_SIZE)

                    if i == 0:
                        buffer = message
                    else:
                        buffer += message

                frame = np.frombuffer(buffer, dtype=np.uint8)
                frame = frame.reshape(frame.shape[0], 1)
                frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                frame = cv2.flip(frame, 1)

                if frame is not None and type(frame) == np.ndarray:
                    #print(f"Frame Recieved")
                    return frame
        return None


    def send_command(self,command):
        message = command
        self.send_message(command)

    def flush_udp(self):
        while True:
            x, _, _ = select.select([self.UDP_CLIENT_SOCKET], [], [], 0.001)
            if len(x) == 0:
                break
            else:
                self.UDP_CLIENT_SOCKET.recv(BUFFER_SIZE)


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

def CheckWhichKeyIsPressed():
    global listener

    if listener == None:
        listener = keyboard.Listener(on_press=on_press, on_release=on_release,suppress=True)
        listener.start()

# Create a listener that runs in the background
#with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
#    listener.join()


if __name__ == '__main__':
    midas = Midas()
    jetson_mock = JetsonMock()
    udp_client = UDPClient()
    CheckWhichKeyIsPressed()
    while True:
        frame = None
        ready = select.select([udp_client.UDP_CLIENT_SOCKET], [], [], 0.2)
        if ready[0]:
            frame = udp_client.receive_frame()
        command =  jetson_mock.get_command()
        if command is None:
            command = 30
        else:
            if frame is not None:
                #print("Received frame")
                #if jetson_mock.czy_przeszkoda(1) is False:
                depth_image = midas.predict(frame)
                jetson_mock.update(depth_image)
            command = jetson_mock.move(command)
            #udp_client.flush_udp()
            if command != 30:
                print(f"Send Command {command}")
                #print("Sending Command")
                udp_client.send_command(command)