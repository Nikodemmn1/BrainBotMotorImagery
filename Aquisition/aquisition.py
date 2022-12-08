import random
import numpy as np
import os
import time


def send_key_to_actiview(impulse):
    key = f"F{impulse+1}"
    os.system(f"xdotool search --name ActiView902-Linux.vi windowactivate key {key}")


number_of_impulse = 20
number_of_impulses = 3 * 20
break_len = 3
impulse_min = 2
impulse_max = 5

impulses = np.concatenate([np.full(number_of_impulse, v) for v in range(1, 4)])
impulses_names = ["BREAK", "LEFT", "RIGHT", "RELAX"]

while True:
    x = input("Press c to continue or q to stop")
    if x == 'q':
        break
    else:
        impulses_randomized = np.copy(impulses)
        np.random.shuffle(impulses_randomized)
        for i, impulse in enumerate(impulses_randomized):
            print(f"BREAK {i+1}/{number_of_impulses}")
            time.sleep(break_len)
            os.system("clear")
            print(impulses_names[impulse])
            send_key_to_actiview(impulse)
            impulse_time = 4
            time.sleep(impulse_time)
            send_key_to_actiview(0)
