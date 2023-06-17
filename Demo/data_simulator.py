import redis
import numpy as np
from redis_utilis import sendRedis
import time
def main():
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    while True:
        data = np.random.random((11, 107))
        print("Sending data:", data.shape)
        sendRedis(r, data, 'eeg_data')
        time.sleep(0.5)

if __name__ == "__main__":
    main()