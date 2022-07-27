from scipy import signal
from parameters import *
import numpy as np

def generate_sequence(t, active_interval, rest_interval):
    return signal.square(2 * np.pi * SQUARE_FREQ * t, duty=DUTY)

def show(side, canvas, left_hand, right_hand, left_foot, right_foot, tongue):
    if (side == 0):
        canvas.itemconfigure(left_hand, state='normal')
        canvas.itemconfigure(right_hand, state='hidden')
        canvas.itemconfigure(left_foot, state='hidden')
        canvas.itemconfigure(right_foot, state='hidden')
        canvas.itemconfigure(tongue, state='hidden')
    elif (side == 1):
        canvas.itemconfigure(left_hand, state='hidden')
        canvas.itemconfigure(right_hand, state='normal')
        canvas.itemconfigure(left_foot, state='hidden')
        canvas.itemconfigure(right_foot, state='hidden')
        canvas.itemconfigure(tongue, state='hidden')
    elif (side == 2):
        canvas.itemconfigure(left_hand, state='hidden')
        canvas.itemconfigure(right_hand, state='hidden')
        canvas.itemconfigure(left_foot, state='normal')
        canvas.itemconfigure(right_foot, state='hidden')
        canvas.itemconfigure(tongue, state='hidden')
    elif (side == 3):
        canvas.itemconfigure(left_hand, state='hidden')
        canvas.itemconfigure(right_hand, state='hidden')
        canvas.itemconfigure(left_foot, state='hidden')
        canvas.itemconfigure(right_foot, state='normal')
        canvas.itemconfigure(tongue, state='hidden')
    elif (side == -1):
        canvas.itemconfigure(left_hand, state='hidden')
        canvas.itemconfigure(right_hand, state='hidden')
        canvas.itemconfigure(left_foot, state='hidden')
        canvas.itemconfigure(right_foot, state='hidden')
        canvas.itemconfigure(tongue, state='hidden')

def start(start_flag):
    start_flag[0] = True
