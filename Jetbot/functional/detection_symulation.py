import threading
import time
import numpy as np

from Jetbot.functional.keyboard_steering import obstacle_detection, command_control, CreateKeyboardListener, get_command, PLOT
from Jetbot.utils.midas_detection import Midas, MidasInterpreter, DecisionMerger
from Jetbot.utils.communication import FrameClient, CommandClient
import Jetbot.utils.configuration  as cfg

SERVER_ADDRESS = ("127.0.0.1", 22243)
JETBOT_ADDRESS = ("127.0.0.1", 22242)
DEBUG_PRINT = False
SYMULATE_COMMANDS = False
PLOT = True

free_boxes = np.array([False, False, False])
free_boxes_lock = threading.Lock()
FRAME_COUNT = 0


def obstacle_detection():
    global free_boxes
    global FRAME_COUNT
    midas = Midas(CAMERA=True)
    midas_iterpreter = MidasInterpreter()
    comm = FrameClient(SERVER_ADDRESS)
    local_frame_count = 0
    frame = None
    while True:
        frame = comm.receive_frame()
        if frame is None:
            continue
        depth_image = midas.predict(frame,PLOT,0,FRAME_COUNT)
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
    if SYMULATE_COMMANDS:
        main()
    else:
        obstacle_detection()