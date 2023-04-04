import select
import cv2
import numpy as np
import socket
import pickle
import math

# Debug mode:
DEBUG_PRINT = False

COMMANDS = {
     0: 'left',
     1: 'right',
     2: 'forward',
}


class FrameClient:
    BUFFER_SIZE = 1024
    MAX_LENGTH_FRAME = 65540

    def __init__(self,server_address):
        print("Connecting to Server...")
        self.SERVER_ADDRESS_PORT = server_address
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

    def __del__(self):
        self.UDP_CLIENT_SOCKET.close()



class CommandClient:
    BUFFER_SIZE = 1024

    def __init__(self, jetbot_address):
        self.JETBOT_ADDRESS_PORT = jetbot_address
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

    def __del__(self):
        self.UDP_CLIENT_SOCKET.close()