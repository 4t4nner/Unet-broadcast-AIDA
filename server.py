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


def check_download():
    if not (path.isfile('model_scripted.pt')):
        subprocess.run(["pip","install", "gdown"])
        subprocess.run(["gdown", "1H3Bd31C8x8GfZinmNLDs-r2RhYa0U7c_"])

dir = path.dirname(path.abspath(__file__))
chdir(dir)

# device = torch.device("cpu")
device = torch.device("cuda")

# model_scripted = torch.jit.script(model) # Export to TorchScript
# model_scripted.save('model_scripted.pt') # Save
model = torch.jit.load('model_scripted.pt')
model.eval()

model.cuda(device)

# what type?
def predict(model, data: cv2.typing.MatLike):
    Y_pred = [model(X_batch) for X_batch in data]
    return np.array(Y_pred)

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
                im = predict(model, decimg)
                cv2.imshow("image", im)
                cv2.waitKey(1)
        except Exception as e:
            print(e)
            self.socketClose()
            cv2.destroyAllWindows()
            self.socketOpen()
            self.receiveThread = threading.Thread(target=self.receiveImages)
            self.receiveThread.start()

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