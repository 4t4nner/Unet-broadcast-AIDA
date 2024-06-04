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

from torch.utils.data import Dataset
import cv2
class SegmentationDataset(Dataset):
    def __init__(self, pathToImages, pathToMasks, imagePaths, maskPaths, transform):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.pathToMasks = pathToMasks
        self.pathToImages = pathToImages
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.transform = transform
    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.imagePaths)
    def __getitem__(self, idx):
        # grab the image path from the current index
        imagePath = self.pathToImages + self.imagePaths[idx]
        # load the image from disk, swap its channels from BGR to RGB,
        # and read the associated mask from disk in grayscale mode
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.pathToMasks + self.maskPaths[idx], 0)
        # check to see if we are applying any transformations
        if self.transform is not None:
            # apply the transformations to both image and its mask
            transformed = self.transform(image=image, mask=mask)
        # return a tuple of the image and its mask
        return (transformed['image'], transformed['mask'])
    
train_transform =  A.Compose(
    [
        A.augmentations.geometric.resize.Resize(720, 720),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RandomCrop(576, 576),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.augmentations.transforms.Normalize()
    ]
)
dataset = SegmentationDataset(".\\data\\imgs\\", ".\\data\\masks\\", os.listdir(".\\data\\imgs\\"), os.listdir(".\\data\\masks\\"), transform=train_transform)

model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=True)

device = torch.device("cuda")
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)
model.cuda(device)
print(device)

learning_rate = 0.001
criterion = nn.MSELoss().cuda(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_total_steps = len(dataloader)
for epoch in range(2):
    for i, (images, masks) in enumerate(dataloader):
        #forward
        # print(images.shape, masks.shape)
        print(i)
        masks = masks.cuda(device)
        # print(masks)
        #.reshape((masks.shape[0], masks.shape[1], 1))
        images = images.cuda(device)
        # print(images)
        images = images.permute(0, 3, 1, 2)
        
        outputs = model(images).cuda(device)
        # print(outputs)
        loss = criterion(outputs, masks.type(torch.Tensor).cuda(device))
        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 2 == 0:
            print(f"epoch: {(epoch+1)}/{2}, step: {i+1}/{n_total_steps}, loss: {loss.item()}")
print("Finished training")


model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save('model_scripted.pt') # Save

# model = torch.jit.load('model_scripted.pt')
# model.eval()