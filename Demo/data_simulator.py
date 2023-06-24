import redis
import numpy as np
from redis_utilis import sendRedis
import time

number_to_label = {
    0: 'left',
    1: 'right',
    2: 'forward',
    3: 'STOP'
}
def main():
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    while True:
        label = np.random.randint(0, 4)
        label = number_to_label[label]
        data = np.random.random((11, 107))
        print("Sending data:", data.shape)
        sendRedis(r, data, 'eeg_data')
        r.set('label', label)
        time.sleep(0.5)

if __name__ == "__main__":
    main()