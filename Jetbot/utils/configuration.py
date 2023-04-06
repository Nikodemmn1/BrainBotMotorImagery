import os

FRAME_SHAPE = (512,512,3) #(384, 384, 3)
FRAME_COUNT = 0

COMMANDS = {
    0: 'left',
    1: 'right',
    2: 'forward',
}
INVCOMMANDS = {
    'left': 0 ,
    'right': 1,
    'forward':2
}

RESOURCES_PATH = '../resources/jetbotimg'

def find_last_img(path=RESOURCES_PATH+'/frame'):
    biggest_num = 0
    files = os.listdir(path)
    if len(files) == 0:
        return 0
    for file in files:
        if file.split('.')[-1] == 'png':
            num = int(file.split('.')[-2].split('_')[-1])
            if biggest_num < num:
                biggest_num = num
    return biggest_num + 1