import socket
import struct
import random
import pickle
from enum import Enum

import Server.server_data_convert as dc
import numpy as np
import torch
import time
import threading
from Server.server_params import *
from Utilities.decision_making import DecisionMaker
from Jetbot.utils.midas_detection import Midas, MidasInterpreter
from Jetbot.utils.communication import FrameClient, CommandClient
from Jetbot.utils.configuration import COMMANDS, INVCOMMANDS
from pynput import keyboard
from Models.OneDNet import OneDNet
from Dataset.dataset import *

import redis

number_to_label_map = {
    '0': 'left',
    '1': 'right',
    '2': 'relax',
    '3': 'neither',
    None: None
}

DATA_PATH = 'DataBDF/Out/KubaBinary/'

EEGKEY = None

JETBOT_ADDRESS = '192.168.193.149'
JETBOT_PORT = 3333

SERVER_ADDRESS = '192.168.193.135'
SERVER_PORT = 22243

COMMAND_SERVING_PORT = 22242

free_boxes = np.array([False, False, False]) # The "resource" that zenazn mentions.
free_boxes_lock = threading.Lock()

KEYS = [False,False,False]
EEGKEY = None
listener = None

class DecisionMerger:
    def __init__(self,print=True):
        self.przeszkoda = 0
        self.print = print

    # Requires usage of a lock with free_boxes_lock:
    def merge(self, command, free_boxes):
        left  = free_boxes[0]
        front = free_boxes[1]
        right = free_boxes[2]

        if command == 'forward':
            if front:
                if self.print:
                    print("Robot jedzie do przodu")
                self.przeszkoda = 0
                return 'forward'
            else:
                if self.print:
                    print("Wykryto przeszkodę Robot stoi")
        elif command == 'left':
            if self.print:
                print("Robot skreca w lewo")
            self.przeszkoda = 0
            return 'left'
        elif command == 'right':
            if self.print:
                print("Robot skreca w prawo")
            self.przeszkoda = 0
            return 'right'
        else:
            return None


def create_sockets():
    # # TCP Socket for receiving data from Actiview
    # tcp_client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # tcp_client_sock.bind(("localhost", TCP_LOCAL_PORT))
    # tcp_client_sock.connect((TCP_AV_ADDRESS, TCP_AV_PORT))

    # UDP socket for sending classification results to the client
    udp_server_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_server_sock.bind((SERVER_ADDRESS, COMMAND_SERVING_PORT))

    return udp_server_sock


def load_mean_std():
    """Loading the file containing the pickled mean and std of the training dataset."""
    with open("../mean_std.pkl", "rb") as mean_std_file:
        mean_std = pickle.load(mean_std_file)
    return mean_std


def load_model():
    """Loading the trained OneDNet model"""
    #model = OneDNet.load_from_checkpoint(checkpoint_path="Modele/Kuba/Multi/model.ckpt", included_classes=[0, 1, 2],
    #                                      channel_count=3)
    model = torch.load("Demo/JointNikodemKuba_multiclass.pt")
    model.eval()
    return model

kierunki = {'0' : 'lewo', '1': 'prawo', '2':'prosto', 'None':'None'}


def obstacle_detection():
    global free_boxes
    midas = Midas(CAMERA=True)
    midas_iterpreter = MidasInterpreter()
    comm = FrameClient((SERVER_ADDRESS,SERVER_PORT))
    frame = None
    while True:
        frame = comm.receive_frame()
        if frame is None:
            continue
        depth_image = midas.predict(frame,False,0,0)
        if depth_image is None:
            continue
        new_free_boxes = midas_iterpreter.find_obstacles(depth_image,print_debug=False)
        if new_free_boxes is None:
            continue
        with free_boxes_lock:
            free_boxes = new_free_boxes

class DataPicker():
    def __init__(self, mode='train'):
        included_classes = [0, 1, 2]
        included_channels = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        full_dataset = EEGDataset("Demo/KubaMulticlass_Train.npy",
                                  "Demo/KubaMulticlass_val.npy",
                                  "Demo/KubaMulticlass_test.npy",
                                  included_classes, included_channels)
        train_dataset, val_dataset, test_dataset = full_dataset.get_subsets()
        if mode=='train':
            labels = train_dataset.dataset.labels[train_dataset.indices]
            data = train_dataset.dataset.data[train_dataset.indices]
        elif mode=='val':
            labels = val_dataset.dataset.labels[val_dataset.indices]
            data = val_dataset.dataset.data[val_dataset.indices]
        elif mode=='test':
            labels = test_dataset.dataset.labels[test_dataset.indices]
            data = test_dataset.dataset.data[test_dataset.indices]
        else:
            raise ValueError("mode must be either 'train', 'val' or 'test'")

        self.dataset = {'data':  torch.from_numpy(data),
                        'labels': torch.from_numpy(labels)}
        self.prepare_queues()

    def __call__(self, label):
        if label not in ['left', 'right', 'relax']:
            #raise ValueError('label should be either left, right or relax')
            return None
        data = self.queues[label][0]
        self.queues[label] = np.roll(self.queues[label], -1, axis=0)
        return torch.Tensor(data).unsqueeze(dim=0)

    def prepare_queues(self):
        labels = self.dataset['labels']
        masks = {'left': labels == 0,
                 'right': labels == 1,
                 'relax': labels == 2}
        self.queues = {'left': self.dataset['data'][masks['left']],
                   'right': self.dataset['data'][masks['right']],
                   'relax': self.dataset['data'][masks['relax']]}


def main():
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)

    udp_server_sock = create_sockets()
    data_picker = DataPicker()
    model = load_model()

    decision_maker = DecisionMaker(window_length=80, priorities=[2, 0, 1], thresholds=[0.55, 0.50, 0.75])
    decisions_to_ignore = 0
    decision_ignored = None
    prev_decision = None

    time_start = time.time()
    seq_num = random.randint(0, 2 ^ 32 - 1)

    # Start a thread to predict on frames
    predicting = threading.Thread(target=obstacle_detection)
    predicting.start()
    decision_merger = DecisionMerger()
    decision_source = 'left'
    is_frame_data : bool = False

    while True:
        number_label = get_eeg_command()
        label = number_to_label_map[number_label]
        if label is not None:
            decision_source = label
        x = data_picker(decision_source)
        if x is not None:
            x = x.cpu().detach().numpy()[0][0][0][0]
            r.xadd('eeg_data', {'data': x})
        if x is not None:
            y = dc.get_classification(x, model)
            out_ind = np.argmax(y.numpy())
            decision_maker.add_data(out_ind)
            is_frame_data = True

        if time.time() - time_start > 0.75:
            # DO the prediction:
            # check if keyboard override
            keyboard_command = get_command()
            if keyboard_command is not None:
                decision = keyboard_command
            else:
                # Check if there is new data to process:
                if is_frame_data:
                    # Detect the EEG signals:
                    decision = str(decision_maker.decide())
                else:
                    # No new data - standby
                    time_start = time.time()
                    is_frame_data = False
                    continue

            is_frame_data = False
            # Check weather ignoring the decision
            if decisions_to_ignore > 0 and prev_decision != decision and decision_ignored != decision:
                # Ignoring
                decisions_to_ignore -= 1
            else:
                # using the decision
                if decision == '0' or decision == '1':
                    decisions_to_ignore = 3
                    decision_ignored = decision
                print(f"EEG Decision: {kierunki[decision]}")
                print(decision_maker.decisions_masks)
                if decision is not None and decision != 'None':
                    with free_boxes_lock:
                        decision = decision_merger.merge(COMMANDS[int(decision)],free_boxes)
                # Sending the control command to the robot
                if decision is not None and decision != 'None':
                    bytes_to_send = str.encode(str(INVCOMMANDS[decision]))
                    udp_server_sock.sendto(bytes_to_send, (JETBOT_ADDRESS, JETBOT_PORT))
            time_start = time.time()
            prev_decision = decision

        seq_num += 1
        if seq_num == 2 ^ 32:
            seq_num = 0

def get_command():
    # Check for arrow key input
    if KEYS[2]:
        #print('Up arrow key pressed')
        return '2'
    elif KEYS[0]:
        #print('Left arrow key pressed')
        return '0'
    elif KEYS[1]:
        #print('Right arrow key pressed')
        return '1'
    return None

def get_eeg_command():
    # Check for 0 1 2 3 key input
    y = EEGKEY
    if EEGKEY is not None:
        #print("Pressed Key " + str(y))
        y = str(y)
    return y

class SignalTypes(Enum):
    Letf = 0
    Straight = 1
    Right = 2
    Neither = 3

def on_press(key):
    global KEYS
    global PLOT
    global EEGKEY
    try:
        if key == keyboard.Key.up:
            #print('Up arrow key pressed')
            KEYS = [False, False, True]
        elif key == keyboard.Key.down:
            #print('Down arrow key pressed')
            KEYS = [False, False, False]
        elif key == keyboard.Key.left:
            #print('Left arrow key pressed')
            KEYS = [True, False, False]
        elif key == keyboard.Key.right:
            #print('Right arrow key pressed')
            KEYS = [False ,True, False]
        elif key == keyboard.Key.f1:
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
    global KEYS
    global PLOT
    global EEGKEY
    try:
        if key == keyboard.Key.up:
            KEYS = [False,False,False]
        elif key == keyboard.Key.left:
            KEYS = [False,False,False]
        elif key == keyboard.Key.right:
            KEYS = [False,False,False]
        elif key == keyboard.Key.f4:
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

    if listener == None:
        listener = keyboard.Listener(on_press=on_press, on_release=on_release,suppress=True)
        #listener.start()

if __name__ == '__main__':
    CreateKeyboardListener()
    main()
