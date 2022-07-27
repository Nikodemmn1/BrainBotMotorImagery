from sqlite3 import connect
from tkinter import *
from parameters import *
import random
import time
import numpy as np
from communication_parameters import *
import communication_modules as cm
import app_modules as am

if __name__ == '__main__':
    window = Tk()

    canvas = Canvas(window, width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
    canvas.pack()
    
    start_flag = [False]
    #times = []
    target = []
    data = []
    left_hand = canvas.create_rectangle(300, 300, 400, 400, fill='gray')
    right_hand = canvas.create_rectangle(800, 300, 900, 400, fill='gray')
    left_foot = canvas.create_rectangle(300, 600, 400, 700, fill='blue')
    right_foot = canvas.create_rectangle(800, 600, 900, 700, fill='blue')
    tongue = canvas.create_rectangle(550, 100, 650, 200, fill='red')

    B = Button(window, text ="START", command =lambda: am.start(start_flag))
    B.pack()

    while(start_flag[0] == False):
        window.update()
        print("waiting to start..")

    sock = cm.connect_to_server(TCP_IP_ADDRESS, TCP_PORT)

    rest_time = True
    t_start = time.time()


    while (time.time() - t_start < DURATION):
        window.update()
        square = am.generate_sequence(time.time(), ACTION_TIME, REST_TIME)
        if (square == -1):
            rest_time = True
            am.show(square, canvas, left_hand, right_hand, left_foot, right_foot, tongue)
            side = -1
            print(side)
        elif(square == 1 and rest_time):
            rest_time = False
            side = random.randint(0, 3)
            am.show(side, canvas, left_hand, right_hand, left_foot, right_foot, tongue)
            print(side)
        elif(square == 1):
            am.show(side, canvas, left_hand, right_hand, left_foot, right_foot, tongue)
            print(side)
        
        #times.append(time.time() - t_start)
        #target.append(side)
        target = np.concatenate([target, np.repeat([[side]], samples,axis=0)])
        data = np.concatenate([data, cm.get_remote_data(sock)], axis = 1)
        #time.sleep(1/SAMPLING_FREQ)
    data = np.transpose(data)
    np.savetxt('target.csv', [p for p in zip(data, target)])
    #window.mainloop()