from pynput import keyboard
import numpy as np
import threading
import time
from Jetbot.utils.midas_detection import Midas, MidasInterpreter, DecisionMerger
from Jetbot.utils.communication import FrameClient, CommandClient
import Jetbot.utils.configuration as cfg

free_boxes = np.array([False, False, False])
free_boxes_lock = threading.Lock()
KEYS = [False,False,False]
PLOT = False
listener = None


# Connection configuration:
SERVER_ADDRESS = ("192.168.0.103", 22243)
JETBOT_ADDRESS = ("192.168.0.105", 3333)

FRAME_COUNT = 0
IMG_ITERATOR = cfg.find_last_img(cfg.RESOURCES_PATH+'/frame')


def obstacle_detection():
    global free_boxes
    global IMG_ITERATOR
    global FRAME_COUNT
    global PLOT
    midas = Midas(CAMERA=True)
    midas_iterpreter = MidasInterpreter()
    comm = FrameClient(SERVER_ADDRESS)
    local_frame_count = 0
    frame = None
    while True:
        frame = comm.receive_frame()
        if frame is None:
            continue
        depth_image = midas.predict(frame,PLOT,IMG_ITERATOR,FRAME_COUNT)
        if depth_image is None:
            continue
        new_free_boxes = midas_iterpreter.find_obstacles(depth_image)
        if new_free_boxes is None:
            continue
        with free_boxes_lock:
            free_boxes = new_free_boxes

def command_control():
    decision_merger = DecisionMerger()
    udp_client = CommandClient(JETBOT_ADDRESS)
    while True:
        time.sleep(0.7)
        command = get_command()
        if command is None: continue
        # if DEBUG_PRINT:
        print(f"Capturing command {command}")
        with free_boxes_lock:
                new_free_boxes = free_boxes
        command = decision_merger.merge(command,new_free_boxes)
        if command is None: continue
        print(f"Sending Command {command}")
        command_index = cfg.INVCOMMANDS[command]
        udp_client.send_command(command_index)


def get_command():
    # Check for arrow key input
    if KEYS[2]:
        print('Up arrow key pressed')
        return 'forward'
    elif KEYS[0]:
        print('Left arrow key pressed')
        return 'left'
    elif KEYS[1]:
        print('Right arrow key pressed')
        return 'right'
    return None


def on_press(key):
    global KEYS
    global PLOT
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
        elif key == keyboard.Key.space:
            PLOT = True
    except AttributeError:
        # Ignore keys that don't have an ASCII representation
        pass


def on_release(key):
    global KEYS
    global PLOT
    try:
        if key == keyboard.Key.up:
            KEYS = [False,False,False]
        elif key == keyboard.Key.left:
            KEYS = [False,False,False]
        elif key == keyboard.Key.right:
            KEYS = [False,False,False]
        elif key == keyboard.Key.space:
            PLOT = False

    except AttributeError:
        # Ignore keys that don't have an ASCII representation
        pass



# Create a keyboard listener that runs in the background
def CreateKeyboardListener():
    global listener

    if listener == None:
        listener = keyboard.Listener(on_press=on_press, on_release=on_release,suppress=True)
        listener.start()



def main():
    print("Setting up server...")
    CreateKeyboardListener()
    # Start a thread to predict on frames
    predicting = threading.Thread(target=obstacle_detection)
    predicting.start()
    time.sleep(1)
    print("Server configured")
    command_control()








if __name__ == '__main__':
    main()
