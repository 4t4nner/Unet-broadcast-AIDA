import threading
import cv2
import socket
import numpy
import time
from datetime import datetime, timezone
import base64

import math
import torch.nn as nn
import segmentation_models_pytorch as smp
import albumentations as A
from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import json
import kwcoco
import pandas as pd
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torch.nn.functional import relu

from torch.utils.data import Dataset
import cv2

from os import path, chdir
import subprocess
import imutils

# убирает рыбий глаз (работает только на камере№2)
# вернет изображение с черными рамками
def fix_fish_eye(img):
    calibration_data = {"K": [[1662.1356630249934, 0.0, 1317.5651863704445],
                              [0.0, 1658.8207636708667, 1002.9834271046615],
                              [0.0, 0.0, 1.0]],
                        "D": [[-0.06601870645064105],
                              [-0.0021117460402409697],
                              [-0.005375920301905082],
                              [0.006445281735768683]]}
    K = np.array(calibration_data['K'])
    D = np.array(calibration_data['D'])
    image_with_border = cv2.copyMakeBorder(img, top=0, bottom=300,
                                           left=0, right=220,
                                           borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    new_K_with_border = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, image_with_border.shape[0:2], np.eye(3), balance=1.0)
    map1_with_border, map2_with_border = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K_with_border, image_with_border.shape[0:2], cv2.CV_16SC2)
    undistorted_img_with_border = cv2.remap(image_with_border, map1_with_border, map2_with_border,
                                            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return undistorted_img_with_border


def revert_fish_eye(undistorted_img, original_shape):
    K = np.array([[original_shape[1], 0, original_shape[1]/2],
                  [0, original_shape[1], original_shape[0]/2],
                  [0, 0, 1]], dtype=np.float32)
    D = np.array([-0.06, 0.02, 0.00, 0.00], dtype=np.float32)

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K, original_shape, cv2.CV_16SC2)
    original_img = cv2.remap(undistorted_img, map1, map2,
                             interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return original_img


def mask_png_artem(img, kernel_sz=2, iterations=5):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0], None, [180], [0, 180])
    print(hist)
    peak = np.argmax(hist)
    tolerance = 50
    lower_bound = (np.array([max(0, peak - tolerance), 80, 50]))
    upper_bound = np.array([min(255, peak + tolerance), 255, 255])
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    kernel_size = kernel_sz
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return closed_mask


def count_white_pixels_in_mask(mask):
    mask_np = np.array(mask)
    white_pixel_count = np.sum(mask_np == 255)
    return white_pixel_count


def pixels_to_cm2(pixel_count, pixel_distance, real_distance):
    pixels_per_cm = pixel_distance / real_distance  # Плотность пикселей (пикселей на сантиметр)
    cm2 = pixel_count / pixels_per_cm**2  # Вычисляем площадь в сантиметрах квадратных
    return cm2


def find_food_img(img):
    angle_rotate = -1.2
    crop_food_y1 = 408
    crop_food_y2 = 931
    crop_top_black = 257
    crop_down_black = 1663
    crop_left_black = 293
    crop_right_black = 2207
    kernel_sz = 3
    iterations = 5
    height_meters = 5.1

    line_cow_out_of_reach_y1 = 498
    line_cow_out_of_reach_y2 = 861

    # фикс рыбьего глаза
    img_fixed_eye = fix_fish_eye(img)
    cv2.imwrite("fix_eye.jpg", img_fixed_eye)

    # поворот изображения на угол
    height, width = img_fixed_eye.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_rotate, 1.0)
    rotated_image = cv2.warpAffine(img_fixed_eye, rotation_matrix, (width, height))

    # кроп чтоб без черных полос
    cropped_black_image = \
        rotated_image[crop_top_black:crop_down_black, crop_left_black:crop_right_black].copy()
    cv2.imwrite("fix_eye_noblack.jpg", cropped_black_image)

    # кроп изображения только на еде
    cropped_food_image = cropped_black_image[crop_food_y1:crop_food_y2, :].copy()
    cv2.imwrite("cropped_food_image.jpg", cropped_food_image)

    # получить маску корма
    mask_food_img = mask_png_artem(cropped_food_image, kernel_sz, iterations)
    cv2.imwrite("mask_food_img.jpg", mask_food_img)

    # накладываем маску на изображение
    top_black_strip = np.zeros((crop_food_y1, cropped_black_image.shape[1]), dtype=np.uint8)
    bottom_black_strip = np.zeros((cropped_black_image.shape[0]-crop_food_y2, cropped_black_image.shape[1]), dtype=np.uint8)
    mask_with_black_strips = np.vstack([top_black_strip, mask_food_img, bottom_black_strip])
    # красный цвет на маске
    mask_with_black_strips_bgr = cv2.cvtColor(mask_with_black_strips, cv2.COLOR_GRAY2BGR)
    red_pixels = np.ones_like(cropped_black_image, dtype=np.uint8) * [0, 0, 255]
    red_regions = red_pixels * (mask_with_black_strips_bgr > 0)
    alpha = 0.5  # Значение от 0 (полностью прозрачный) до 1 (полностью непрозрачный)
    cropped_black_image_uint8 = cropped_black_image.astype(np.uint8)
    result_img = cropped_black_image_uint8 + alpha * red_regions
    cv2.imwrite("masked_img.jpg", result_img)

    # рисуем дополнительные линии
    # корм сверху
    cv2.line(
        result_img,
        (0, crop_food_y1),
        (result_img.shape[1], crop_food_y1),
        (0, 255, 255),
        1)
    cv2.line(
        result_img,
        (0, line_cow_out_of_reach_y1),
        (result_img.shape[1], line_cow_out_of_reach_y1),
        (0, 255, 0),
        2)

    # корм снизу
    cv2.line(
        result_img,
        (0, line_cow_out_of_reach_y2),
        (result_img.shape[1], line_cow_out_of_reach_y2),
        (0, 255, 0),
        2)
    cv2.line(
        result_img,
        (0, crop_food_y2),
        (result_img.shape[1], crop_food_y2),
        (0, 255, 255),
        1)

    cv2.imwrite("masked+lines.jpg", result_img)

    # посчитать кол-во корма в м^2
    s_all = count_white_pixels_in_mask(mask_food_img)
    s_up = count_white_pixels_in_mask(mask_with_black_strips[crop_food_y1:line_cow_out_of_reach_y1])
    s_down = count_white_pixels_in_mask(mask_with_black_strips[line_cow_out_of_reach_y2:crop_food_y2])
    height_px, width_px = mask_food_img.shape[:2]
    square_s_all = pixels_to_cm2(s_all, height_px, height_meters)
    square_s_up = pixels_to_cm2(s_up, height_px, height_meters)
    square_s_down = pixels_to_cm2(s_down, height_px, height_meters)

    # пишем на изображении площадь
    cv2.putText(
        result_img,
        f"Food all: {round(square_s_all, 3)} m^2",
        (result_img.shape[1]//2-50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 0, 255),
        3)

    cv2.putText(
        result_img,
        f"Food up: {round(square_s_up, 3)} m^2",
        (result_img.shape[1]//2-70, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 0, 255),
        3)

    cv2.putText(
        result_img,
        f"Food down: {round(square_s_down, 3)} m^2",
        (result_img.shape[1]//2-85, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 0, 255),
        3)

    cv2.imwrite("done.jpg", result_img)

    return result_img, (square_s_all, square_s_up, square_s_down)



def check_download():
    if not (path.isfile('model_scripted.pt')):
        subprocess.run(["pip","install", "gdown"])
        subprocess.run(["gdown", "1H3Bd31C8x8GfZinmNLDs-r2RhYa0U7c_"])

dir = path.dirname(path.abspath(__file__))
chdir(dir)

# device = torch.device("cpu")
# device = torch.device("cuda")

# model_scripted = torch.jit.script(model) # Export to TorchScript
# model_scripted.save('model_scripted.pt') # Save
# model = torch.jit.load('model_scripted.pt')
# model.eval()

# model.cuda(device)

# what type?

class ServerSocket:

    def __init__(self, ip, port):
        self.TCP_IP = ip
        self.TCP_PORT = port
        self.socketOpen()
        self.receiveThread = threading.Thread(target=self.receiveImages)
        self.receiveThread.start()

    def socketClose(self):
        self.sock.close()
        print(u'Server socket [ TCP_IP: ' + self.TCP_IP + ', TCP_PORT: ' + str(self.TCP_PORT) + ' ] is close')

    def socketOpen(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.TCP_IP, self.TCP_PORT))
        self.sock.listen(1)
        print(u'Server socket [ TCP_IP: ' + self.TCP_IP + ', TCP_PORT: ' + str(self.TCP_PORT) + ' ] is open')
        self.conn, self.addr = self.sock.accept()
        print(u'Server socket [ TCP_IP: ' + self.TCP_IP + ', TCP_PORT: ' + str(self.TCP_PORT) + ' ] is connected with client')

    def receiveImages(self):
        cv2.startWindowThread()

        try:
            while True:
                length = self.recvall(self.conn, 64)
                length1 = length.decode('utf-8')
                stringData = self.recvall(self.conn, int(length1))
                stime = self.recvall(self.conn, 64)
                print('send time: ' + stime.decode('utf-8'))
                now = time.localtime()
                print('receive time: ' + datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f'))
                data = numpy.frombuffer(base64.b64decode(stringData), numpy.uint8)
                decimg = cv2.imdecode(data, 1)
                cv2.imshow('orig', decimg)
                cv2.waitKey(1)
                img = cv2.resize(decimg, (2592, 1944),interpolation=cv2.INTER_CUBIC)
                im = find_food_img(img)[0]
                
                cv2.imwrite('test.jpg', im)
                cv2.imshow("pred", im)

                k = cv2.waitKey(0)
                print(k)
                imr = cv2.imread('test.jpg')
                cv2.imshow("imr", imr)
                k = cv2.waitKey(1)
                
                if k == ord('q'):
                    print('quit')
                    break
        except Exception as e:
            print(e)
            self.socketClose()
            cv2.destroyAllWindows()
            self.socketOpen()
            self.receiveThread = threading.Thread(target=self.receiveImages)
            self.receiveThread.start()
        cv2.destroyAllWindows()

    def recvall(self, sock, count):
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf

def main():
    dir = path.dirname(path.abspath(__file__))
    chdir(dir)
    server = ServerSocket('localhost', 8081)
    


if __name__ == "__main__":
    main()