import cv2
import socket
import math
import pickle
import sys
from PIL import Image
import time

max_length = 65000
host = "127.0.0.1"
port = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

time.sleep(0.2)
#cap = cv2.VideoCapture(0)
#ret, frame = cap.read()

path = "../img/"
frames = ["girl_Gertruda.png","girl_Cukierek.png","girl_Kalwina.jpg","girl_Korona_86.jpg","girl_Milana_11.jpg"
    ,"girl_Mizara.png","girl_Nastia_64.jpg"]

for _ in range(5):
    frames.extend(frames)

for img in frames:
    # compress frame
    print(f"Sending frame {img}")
    frame = cv2.imread(path+img)
    retval, buffer = cv2.imencode(".jpg", frame)

    if retval:
        # convert to byte array
        buffer = buffer.tobytes()
        # get size of the frame
        buffer_size = len(buffer)

        num_of_packs = 1
        if buffer_size > max_length:
            num_of_packs = math.ceil(buffer_size / max_length)

        frame_info = {"packs": num_of_packs}

        # send the number of packs to be expected
        print("Number of packs:", num_of_packs)
        sock.sendto(pickle.dumps(frame_info), (host, port))

        left = 0
        right = max_length

        for i in range(num_of_packs):
            print("left:", left)
            print("right:", right)

            # truncate data to send
            data = buffer[left:right]
            left = right
            right += max_length

            # send the frames accordingly
            sock.sendto(data, (host, port))


print(f"Send {len(frames)} frames, done")