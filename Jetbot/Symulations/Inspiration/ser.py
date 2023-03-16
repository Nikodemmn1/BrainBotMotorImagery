import cv2
import socket
import pickle
import numpy as np

host = "127.0.0.1"
port = 5005
max_length = 65540

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((host, port))

frame_info = None
buffer = None
frame = None

print("-> waiting for connection")

count=0
while True:
    data, address = sock.recvfrom(max_length)

    if len(data) < 100:
        frame_info = pickle.loads(data)

        if frame_info:
            nums_of_packs = frame_info["packs"]
            print(f"New Frame with {nums_of_packs} pack(s)")

            for i in range(nums_of_packs):
                data, address = sock.recvfrom(max_length)

                if i == 0:
                    buffer = data
                else:
                    buffer += data

            frame = np.frombuffer(buffer, dtype=np.uint8)
            frame = frame.reshape(frame.shape[0], 1)

            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            frame = cv2.flip(frame, 1)
            #frame = np.flip(frame, axis=1)

            if frame is not None and type(frame) == np.ndarray:
                print(f"Frame {count} Recieved")

                # cv2.imshow("Stream", frame)
                # if cv2.waitKey(1) == 27:
                #     break
    count += 1

print("goodbye")