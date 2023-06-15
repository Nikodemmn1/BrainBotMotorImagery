import time
import os
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
from Server.server_params import *
import socket
from tqdm import tqdm
from pynput import keyboard

EEGKEY = None
listener = None


class SignalTypes(Enum):
    Letf = 0
    Straight = 1
    Right = 2
    Neither = 3

def load_data(raw_path, save_path):
    CHANNELS_TO_SEND = CHANNELS
    # raw_path:
    # usually "DataBDF/file.bdf"
    # or "DataBDF/' then all '.bdf' files in dir
    # save_path - np: "testdata.npy"
    # https://www.biosemi.com/faq/file_format.htm
    CHANNELS_IN_FILE = 17  # with triggers
    HEADER_LENGTH = 256 * (CHANNELS_IN_FILE + 1)
    SAMPLING_RATE = 2048

    FILE_PATHS = []
    dir_path = '/'.join(raw_path.split("/")[:-1])+'/'
    if os.path.isdir(raw_path):
        FILE_PATHS = os.listdir(raw_path)
    elif os.path.isfile(raw_path):
        file_path = raw_path.split("/")[-1]
        FILE_PATHS = [file_path]
    else:
        print("The raw path does not exist.")
        return None

    samples_list = []
    for file_path in FILE_PATHS:
        path = dir_path + file_path
        file_bytes = os.stat(path).st_size
        file_bytes_no_head = file_bytes - HEADER_LENGTH

        channel_sections_count = file_bytes_no_head // (CHANNELS_IN_FILE * SAMPLING_RATE * 3)

        with open(path, 'rb') as f:
            data = f.read()
        data = np.frombuffer(data[HEADER_LENGTH:], dtype='<u1')

        samples = np.ndarray((CHANNELS_TO_SEND, SAMPLING_RATE * channel_sections_count, 3), dtype='<u1')

        for sec in tqdm(range(channel_sections_count)):
            for ch in range(CHANNELS_TO_SEND):
                for sam in range(SAMPLING_RATE):
                    beg = sec * CHANNELS_IN_FILE * SAMPLING_RATE * 3 + ch * SAMPLING_RATE * 3 + sam * 3
                    samples[ch, sec * SAMPLING_RATE + sam, :] = data[beg:beg + 3]
        samples_list.append(samples)
    samples_to_save = np.concatenate(samples_list, axis = 1)
    np.save(save_path, samples_to_save)
    print("Data Saved to Numpy")


class EEGSignals:
    files = ['testdata.npy','testdata.npy','testdata.npy','testdata.npy']

    def __init__(self,signal_type : SignalTypes):
        x = np.load(self.files[signal_type.value])
        x = np.transpose(x, (1, 0, 2)).flatten()
        self.packets_data = np.reshape(x, (-1, WORDS * 3))
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < self.packets_data.shape[0]:
            return self.packets_data[self.index, :].tobytes()
            self.index += 1
        else:
            raise StopIteration

    def __len__(self):
        return self.packets_data.shape[0]

def main_idea():
    eeg_signal_type = SignalTypes.Neither
    eeg_signal = EEGSignals(eeg_signal_type)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("localhost", TCP_AV_PORT))
    sock.listen(0)
    conn, addr = sock.accept()
    with conn:
        print("Connected!")
        while(True):
            try:
                signal = next(iter(eeg_signal))
            except StopIteration:
                eeg_signal = iter(eeg_signal)
                continue

            conn.setblocking(True)
            conn.send(signal)
            time.sleep(0.0625)

            # Check if new orders come in:
            conn.setblocking(False)
            # Perform non-blocking receive
            received_data = conn.recv(4096)
            if received_data:
                # Load EEG File again
                received_data = received_data.decode()
                eeg_signal = EEGSignals(SignalTypes(received_data))

def main():
    load_data('Data/Kuba_Raw/kub2_3.bdf','Data/kub2_3.npy')
    eeg_signal_type = SignalTypes.Neither
    eeg_signal = EEGSignals(eeg_signal_type)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("localhost", TCP_AV_PORT))
    sock.listen(0)
    conn, addr = sock.accept()
    with conn:
        print("Connected!")
        while(True):
            try:
                signal = next(iter(eeg_signal))
            except StopIteration:
                eeg_signal = iter(eeg_signal)
                print("Magazine Empty")
                continue

            conn.send(signal)
            time.sleep(0.0625)

            # Check if new orders come in:
            eeg_command = get_eeg_command()
            if eeg_command is not None:
                # Load EEG File again
                eeg_signal = EEGSignals(SignalTypes(eeg_command))
                print("Changed Signals to " + str(eeg_command))

def get_eeg_command():
    # Check for 0 1 2 3 key input
    if EEGKEY is not None:
        print("Pressed Key " + str(EEGKEY))
    return EEGKEY

def on_press(key):
    global EEGKEY
    try:
        if key == keyboard.Key.f1:
            EEGKEY = 0
        elif key == keyboard.Key.f2:
            EEGKEY = 1
        elif key == keyboard.Key.f3:
            EEGKEY = 2
        elif key == keyboard.Key.f4:
            EEGKEY = 3

    except AttributeError:
        # Ignore keys that don't have an ASCII representation
        pass


def on_release(key):
    global EEGKEY
    try:
        if key == keyboard.Key.f4:
            EEGKEY = None
        elif key == keyboard.Key.f1:
            EEGKEY = None
        elif key == keyboard.Key.f2:
            EEGKEY = None
        elif key == keyboard.Key.f3:
            EEGKEY = None


    except AttributeError:
        # Ignore keys that don't have an ASCII representation
        pass

# Create a keyboard listener that runs in the background
def CreateKeyboardListener():
    global listener
    #if listener == None:
        #listener = keyboard.Listener(on_press=on_press, on_release=on_release,suppress=True)
        #listener.start()

if __name__ == '__main__':
    CreateKeyboardListener()
    main()