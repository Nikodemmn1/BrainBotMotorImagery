import nanocamera as nano

# To musi być na początku, bo inaczej wywali błąd
RESOLUTION = (384, 384)
camera = nano.Camera(flip=0, width=RESOLUTION[0], height=RESOLUTION[1], fps=10)
frame = camera.read()
print('Pierwszy frame', frame)

import asyncio
import torch
import urllib.request
import os
import cv2
import time
import numpy as np
from jetbot import Robot

MEAN_PIXEL_COUNT_RATIO = 0.1
MEAN_PIXEL_COUNT = int(RESOLUTION[0] * 0.35 * RESOLUTION[1] * 0.2 * MEAN_PIXEL_COUNT_RATIO)
Y_BOX_POSITION = (int(RESOLUTION[1] * 0.4), int(RESOLUTION[1] * 0.6))
X_BOX_POSITION = (
    int(RESOLUTION[0] * 0.05), int(RESOLUTION[0] * 0.35), int(RESOLUTION[0] * 0.7), int(RESOLUTION[0] * 0.95))
frame_count = 3
min_safe_distance = 500


def load_model_midas():
    print('load_midas')
    # model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    # model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    return midas, transform, device


def predict_midas(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    return prediction.cpu().numpy()


async def move_robot(is_free, speed=0.3, sleep_time=0.2):
    if is_free[1] and is_free[0] and is_free[2]:
        print('FORWARD')
        robot.forward(speed)
    elif is_free[1] and is_free[0]:
        print('LEFT')
        robot.left(speed)
        time.sleep(0.1)
        robot.forward(speed)
    elif is_free[1] and is_free[2]:
        print('RIGHT')
        robot.left(speed)
        time.sleep(0.1)
        robot.forward(speed)
    elif is_free[0]:
        print('BACK LEFT')
        robot.left(speed)
        time.sleep(sleep_time)
        robot.forward(speed)
    elif is_free[2]:
        print('BACK RIGHT')
        robot.right(speed)
        time.sleep(sleep_time)
        robot.forward(speed)
    else:
        print('BACK AROUND')
        robot.backward(speed)
        time.sleep(0.3)
        robot.left(speed)

    time.sleep(sleep_time)
    robot.stop()


def save_frame(frame):
    cv2.imwrite(f'./images/image_{i}.jpg', frame[120:160])
    return 1


def mean_biggest_values(array):
    array = array.flatten()
    ind = np.argpartition(array, -MEAN_PIXEL_COUNT)[MEAN_PIXEL_COUNT:]
    return np.average(array[ind])


def average(depth_image):
    left = mean_biggest_values(depth_image[Y_BOX_POSITION[0]:Y_BOX_POSITION[1], X_BOX_POSITION[0]:X_BOX_POSITION[1]])
    mid = mean_biggest_values(depth_image[Y_BOX_POSITION[0]:Y_BOX_POSITION[1], X_BOX_POSITION[1]:X_BOX_POSITION[2]])
    right = mean_biggest_values(depth_image[Y_BOX_POSITION[0]:Y_BOX_POSITION[1], X_BOX_POSITION[2]:X_BOX_POSITION[3]])
    return np.array([left, mid, right])


def get_free_boxes(avg):
    avg /= frame_count
    is_free = avg < min_safe_distance

    print(f"Distances: left={avg[0]} middle={avg[1]} right{avg[2]}")
    print(f"is_free: left={is_free[0]} middle={is_free[1]} right{is_free[2]}")

    return is_free


if __name__ == '__main__':
    robot = Robot()
    print('CSI Camera ready? - ', camera.isReady())
    midas, transform, device = load_model_midas()

    avg = np.array([0., 0., 0.])
    i = 0
    while True:
        i += 1
        frame = camera.read()
        depth_image = predict_midas(frame)
        avg += average(depth_image)

        if i == frame_count:
            is_free = get_free_boxes(avg)
            asyncio.run(move_robot(is_free))
            avg = np.array([0., 0., 0.])
            i = 0

    robot.stop()
    camera.release()
    del camera
