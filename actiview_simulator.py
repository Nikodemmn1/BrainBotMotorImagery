import time
from Server.server_params import *
import socket

buff = bytearray(WORDS*3)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((TCP_AV_ADDRESS, TCP_AV_PORT))
sock.listen()
conn, addr = sock.accept()

with conn:
    print("Connected!")
    while True:
        conn.send(buff)
        time.sleep(0.0625)
