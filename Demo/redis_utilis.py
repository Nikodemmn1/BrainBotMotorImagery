import numpy as np
import struct
import redis
def sendRedis(database, array, stream_name):
    h,w = array.shape
    shape = struct.pack('>II', h, w)
    encoded = shape + array.tobytes()

    database.set(stream_name, encoded)
    return

def readRedis(database, stream_name):
    data_struct = database.get(stream_name)
    h, w = struct.unpack('>II', data_struct[:8])
    a = np.frombuffer(data_struct[8:]).reshape(h, w)
    return a

if __name__ == "__main__":
    r = redis.Redis(host='localhost', port=6379, decode_responses=False)
    while True:
        data = readRedis(r, 'eeg_data')
        print("Received data", data.shape)

