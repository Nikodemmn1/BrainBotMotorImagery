import nanocamera as nano

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


def load_model_midas():
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


MEAN_PIXEL_COUNT_RATIO = 0.1
MEAN_PIXEL_COUNT = int(RESOLUTION[0] * 0.35 * RESOLUTION[1] * 0.2 * MEAN_PIXEL_COUNT_RATIO)


def mean_biggest_values(array):
    array = array.flatten()
    ind = np.argpartition(array, -MEAN_PIXEL_COUNT)[MEAN_PIXEL_COUNT:]
    return np.average(array[ind])


Y_AVG = (int(RESOLUTION[1] * 0.4), int(RESOLUTION[1] * 0.6))
X_AVG = (int(RESOLUTION[0] * 0.05), int(RESOLUTION[0] * 0.35), int(RESOLUTION[0] * 0.7), int(RESOLUTION[0] * 0.95))


def average(depth_image):
    left = mean_biggest_values(depth_image[Y_AVG[0]:Y_AVG[1], X_AVG[0]:X_AVG[1]])
    mid = mean_biggest_values(depth_image[Y_AVG[0]:Y_AVG[1], X_AVG[1]:X_AVG[2]])
    right = mean_biggest_values(depth_image[Y_AVG[0]:Y_AVG[1], X_AVG[2]:X_AVG[3]])
    return [left, mid, right]


if __name__ == '__main__':
    i = 0
    print('start')
    robot = Robot()
    print('CSI Camera ready? - ', camera.isReady())

    print('load_midas')
    midas, transform, device = load_model_midas()
    avg = [0, 0, 0]
    is_free = [0, 0, 0]
    frame_count = 3
    min_safe_distance = 500

    while True:
        i += 1
        frame = camera.read()

        depth_image = predict_midas(frame)
        avg_temp = average(depth_image)
        avg[0] += avg_temp[0]
        avg[1] += avg_temp[1]
        avg[2] += avg_temp[2]
        if i == frame_count:
            avg[0] = avg[0] / frame_count
            avg[1] = avg[1] / frame_count
            avg[2] = avg[2] / frame_count
            is_free[0] = avg[0] < min_safe_distance
            is_free[1] = avg[1] < min_safe_distance
            is_free[2] = avg[2] < min_safe_distance

            print('x', avg[0], avg[1], avg[2], is_free)
            i = 0
            avg = [0, 0, 0]
            asyncio.run(move_robot(is_free))

    robot.stop()
    camera.release()
    del camera
