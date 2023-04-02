import socket
import select
import pickle
import cv2
import numpy as np

DEBUG_PRINT = False

class FrameClient:
    BUFFER_SIZE = 1024
    DATAGRAM_MAX_SIZE = 65540

    def __init__(self, server_address_port):
        self.SERVER_ADDRESS_PORT = server_address_port
        self.UDP_CLIENT_SOCKET = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.UDP_CLIENT_SOCKET.bind(self.SERVER_ADDRESS_PORT)

    def receive_frame(self):
        message = self.UDP_CLIENT_SOCKET.recv(self.DATAGRAM_MAX_SIZE)
        if len(message) < 100:
            frame_info = pickle.loads(message)
            if frame_info:
                nums_of_packs = frame_info["packs"]
                if DEBUG_PRINT:
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

                if frame is not None and type(frame) == np.ndarray:
                    if DEBUG_PRINT:
                        print(f"Frame Received ")
                    return frame
        return None

    def active_wait(self,seconds=0.2):
        ready = select.select([self.UDP_CLIENT_SOCKET], [], [], seconds)
        return ready[0]


class CommandClient:
    def __init__(self, jetbot_address_port):
        self.JETBOT_ADDRESS_PORT = jetbot_address_port
        self.UDP_CLIENT_SOCKET = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

    def send_command(self, command):
        message = str.encode(str(command))
        self.UDP_CLIENT_SOCKET.sendto(message, self.JETBOT_ADDRESS_PORT)