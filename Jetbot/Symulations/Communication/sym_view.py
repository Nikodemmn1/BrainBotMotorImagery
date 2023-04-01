import numpy as np
import socket

WORDS = 16 * 128
TCP_AV_ADDRESS = '192.168.0.163'
TCP_AV_PORT = 8188  # port configured in Activeview

samples = np.load("../testdata.npy")
samples = np.transpose(samples, (1, 0, 2, 3)).flatten()
packets_data = np.reshape(samples, (-1, WORDS * 3))


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((TCP_AV_ADDRESS, TCP_AV_PORT))
sock.listen(0)
conn, addr = sock.accept()

with conn:
    print("Connected!")
    for i in range(packets_data.shape[0]):
        conn.send(packets_data[i, :].tobytes())
        time.sleep(0.0625)