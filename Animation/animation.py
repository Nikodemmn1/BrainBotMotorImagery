from tkinter import *
import time
from PIL import Image, ImageTk
import socket
import json
from communication_parameters import *

udpClient_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udpClient_sock.bind((UDP_ADDRESS, UDP_PORT))


WIDTH = 1200
HEIGHT = 1200
xVelocity = 5

window = Tk()

canvas = Canvas(window, width=WIDTH, height=HEIGHT)
canvas.pack()

carImage = Image.open('static/car.png')
carImage = carImage.resize((200, 200))
car = ImageTk.PhotoImage(carImage)


my_image = canvas.create_image(WIDTH/2, HEIGHT/2, image=car, anchor=CENTER)

while True:
    data, addr = udpClient_sock.recvfrom(4096)
    result = data.decode("utf-8")
    result_dict = json.loads(result)
    coordinates = canvas.coords(my_image)
    if(coordinates[0] < WIDTH and coordinates[0] > 0):
        if (result_dict['left'] == True):
            canvas.move(my_image, -xVelocity, 0)
        else:
            canvas.move(my_image, xVelocity, 0)
    window.update()
    time.sleep(0.01)


window.mainloop()
