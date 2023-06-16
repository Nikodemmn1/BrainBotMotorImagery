import redis
import numpy as np

def main():
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    while True:
        data = np.random.randint(0, 10)
        print("Sending data:", data)
        r.xadd('eeg_data', {'data': data})

if __name__ == "__main__":
    main()