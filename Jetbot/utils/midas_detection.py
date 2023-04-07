import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import Jetbot.utils.configuration as cfg

class Midas:
    def __init__(self, CAMERA=False):
        midas, transform, device = self.load_model_midas()
        self.midas = midas
        self.transform = transform
        self.device = device
        self.CAMERA = CAMERA

    def load_model_midas(self):
        print('load_midas')
        # model_t
        model_type = "DPT_BEiT_L_512" # MiDaS v3.1 - Large (For highest quality - 3.2023)
        #model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        # model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
        # model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        midas.to(device)
        midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            transform = midas_transforms.dpt_transform
        elif model_type == "DPT_BEiT_L_512":
            transform = midas_transforms.beit512_transform
        else:
            transform = midas_transforms.small_transform
        return midas, transform, device

    def predict(self, img, plot : bool, img_iterator : int, frame_count : int):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)
        with torch.no_grad():
            prediction = self.midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        print(f"Frame predicted {frame_count}")
        frame_count += 1
        if plot:
            plt.figure()
            plt.imshow(img)
            plt.axis('off')
            plt.imsave(f"{cfg.RESOURCES_PATH}/frame/frame_{img_iterator}.png",img)
            plt.show()

            plt.figure()
            plt.imshow(prediction.cpu().numpy())
            plt.axis('off')
            plt.imsave(f"{cfg.RESOURCES_PATH}/pred/pred_{img_iterator}.png",prediction.cpu().numpy())
            plt.axhline(10,0,1,color='black')
            plt.axhline(300,0,1,color='black')
            plt.axvline(52, 0, 1, color='black')
            plt.axvline(112, 0, 1, color='black')
            plt.axvline(272, 0, 1, color='black')
            plt.axvline(332, 0, 1, color='black')
            plt.show()

            np.save(f"{cfg.RESOURCES_PATH}/np/np_{img_iterator}", prediction.cpu().numpy())

            img_iterator += 1
        if self.CAMERA:
            pred = prediction.cpu().numpy()
            pred = (255 * (pred - pred.min()) / (pred.max() - pred.min())).astype(np.uint8)
            cv2.imshow('WebCam', pred)
            cv2.waitKey(1)
            if cv2.waitKey(25) == ord('q'):
                return prediction.cpu().numpy()
        return prediction.cpu().numpy()


class MidasInterpreter:
    # Params for DPT_BEiT_L_512:
    MEAN_SAFE_DISTANCE = 7400  #7250   #20 #7000
    MAX_SAFE_DISTANCE  = 7500  #7500  #25 #7500
    MIN_SAFE_DISTANCE  = 7250  # 7000 # 21  #6000
    GROUP_COUNT = 25
    BOK_GROUP = 7

    GROUP_SIZE = 10
    RESOLUTION = (384,384) #(384, 384)
    MEAN_PIXEL_COUNT_RATIO = 0.1
    MEAN_PIXEL_COUNT = int(RESOLUTION[0] * 0.35 * RESOLUTION[1] * 0.2 * MEAN_PIXEL_COUNT_RATIO)
    Y_BOX_POSITION = (10, 300) #230)#330) # split into 10 - 320 - 54
    X_BOX_POSITION = (52, 112, 272, 332) # split into 22 - 90 - 160 - 90 - 22

    def __init__(self):
        self.free_boxes = np.array([False, False, False])

    def find_obstacles(self,depth_image):
        self.free_boxes = np.array([False, False, False])
        left_part = depth_image[self.Y_BOX_POSITION[0]:self.Y_BOX_POSITION[1], self.X_BOX_POSITION[0]:self.X_BOX_POSITION[1]]
        mid_part = depth_image[self.Y_BOX_POSITION[0]:self.Y_BOX_POSITION[1], self.X_BOX_POSITION[1]:self.X_BOX_POSITION[2]]
        right_part = depth_image[self.Y_BOX_POSITION[0]:self.Y_BOX_POSITION[1], self.X_BOX_POSITION[2]:self.X_BOX_POSITION[3]]

        left_depth, left_count, left_width_count = self.look_for_grouping(left_part)
        mid_depth, mid_count, mid_width_count = self.look_for_grouping(mid_part)
        right_depth, right_count, right_width_count = self.look_for_grouping(right_part)

        left_columns_count = self.look_for_column(left_part)
        mid_columns_count = self.look_for_column(mid_part)
        right_columns_count = self.look_for_column(right_part)

        left_average =  self.mean_biggest_values(left_part)
        mid_average =  self.mean_biggest_values(mid_part)
        right_average =  self.mean_biggest_values(right_part)

        left_free = (left_depth < self.MAX_SAFE_DISTANCE or left_columns_count < self.BOK_GROUP)  and left_count < self.GROUP_COUNT and left_average < self.MEAN_SAFE_DISTANCE
        mid_free = mid_depth < self.MAX_SAFE_DISTANCE and mid_count < self.GROUP_COUNT and mid_average < self.MEAN_SAFE_DISTANCE
        right_free = (right_depth < self.MAX_SAFE_DISTANCE or right_columns_count < self.BOK_GROUP) and right_count < self.GROUP_COUNT and right_average < self.MEAN_SAFE_DISTANCE

        print(f"Depth prediction:\n"
              f"Mean Distances: left={left_average} middle={mid_average} right{right_average}\n"
              f"Max Distances: left={left_depth} middle={mid_depth} right{right_depth}\n"
              f"Distances Count: left={left_count} middle={mid_count} right{right_count}\n"
              f"Width Count: left={left_width_count} middle={mid_width_count} right{right_width_count}\n"
              f"Columns Count: left={left_columns_count} middle={mid_columns_count} right{right_columns_count}\n"
              f"is_free: left={left_free} middle={mid_free} right{right_free}")

        self.free_boxes = np.array([left_free, mid_free, right_free])
        return self.free_boxes


    @staticmethod
    def look_for_grouping(array):
        best_mean = 0.0
        count = 0
        y_count = 0
        y_bool = False
        for x in range(0,array.shape[0],MidasInterpreter.GROUP_SIZE):
            y_bool = False
            for y in range(0,array.shape[1],MidasInterpreter.GROUP_SIZE):
                grid = array[x:x+MidasInterpreter.GROUP_SIZE, y:y+MidasInterpreter.GROUP_SIZE]
                mean = grid.mean()
                if mean > MidasInterpreter.MIN_SAFE_DISTANCE:
                    count += 1
                    if y_bool is False:
                        y_count +=1
                        y_bool = True
                if mean > best_mean:
                    best_mean = mean

        return best_mean, count, y_count

    @staticmethod
    def look_for_column(array):
        columns = 0
        for y in range(0, array.shape[1], MidasInterpreter.GROUP_SIZE):
            x_count = 0
            for x in range(0, array.shape[0], MidasInterpreter.GROUP_SIZE):
                grid = array[x:x + MidasInterpreter.GROUP_SIZE, y:y + MidasInterpreter.GROUP_SIZE]
                mean = grid.mean()
                if mean > MidasInterpreter.MIN_SAFE_DISTANCE:
                    x_count += 1
                    if x_count == 1:
                        columns += 1
        return columns




    @staticmethod
    def mean_biggest_values(array):
        pixel_count = int(array.shape[0]* array.shape[1] * 0.1)
        array = array.flatten()
        ind = np.argpartition(array, - pixel_count)[pixel_count:]
        return np.average(array[ind])


class DecisionMerger:
    def __init__(self):
        self.przeszkoda = 0

    # Requires usage of a lock with free_boxes_lock:
    def merge(self, command, free_boxes):
        left  = free_boxes[0]
        front = free_boxes[1]
        right = free_boxes[2]

        if command == 'forward':
            # if self.przeszkoda > 0:
            #     print(f"Przeszkoda czekam midas_detection {self.przeszkoda}")
            #     self.przeszkoda -= 1
            #     return None
            # if front and left and right:
            if front:
                print("Robot jedzie do przodu")
                self.przeszkoda = 0
                return 'forward'
            elif left and not front and not right:
                return 'left'
            elif right and not front and not left:
                return 'right'
            elif left and front and not right:
                return 'front'
            elif right and front and not left:
                return 'front'
            else:
                # return random.choice(['left', 'right'])
                return 'right'
                # print("Przeszkoda akcja nie jest podjęta!! 909090909090909090909090909090909090")
                # self.przeszkoda = 3
                # return None
        elif command == 'left':
            print("Robot skreca w lewo")
            self.przeszkoda = 0
            return 'left'
        elif command == 'right':
            print("Robot skreca w prawo")
            self.przeszkoda = 0
            return 'right'
        else:
            return None