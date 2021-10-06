from __future__ import print_function, division
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import cv2


class CrackDataset(Dataset):
    def __init__(self, data_path, phase='train', transform=None):
        self.data_path = data_path
        self.phase = phase
        self.transform = transform
        filepath = os.path.join(self.data_path,phase)
        file_list = os.listdir(filepath)
        self.data = []
        for class_name in file_list:
            class_dir = os.path.join(filepath, class_name)
            list_inside_dir = os.listdir(class_dir)
            for item in list_inside_dir:
                img_path = os.path.join(class_dir, item)
                self.data.append([img_path,class_name])
        self.class_map = {'Negative':0, 'Positive': 1}
        # self.img_dim = (64,64)


    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        img_path, class_name = self.data[idx]
        img = Image.open(img_path).convert("RGB") # channel , width, height
        if self.transform is not None:
            img_tensor = self.transform(img)
        class_id = self.class_map[class_name]
        class_id = torch.tensor([class_id])
        return img_tensor.float(), class_id


if __name__ == "__main__":
    data_path = 'data/'
    train_data = CrackDataset(data_path, 'train')
    test_data = CrackDataset(data_path, 'test')

    traindata_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    testdata_loader = DataLoader(test_data, batch_size=32, shuffle=True)

    # for imgs, labels in traindata_loader:
    #     print("Batch of images has shape: ",imgs.shape)
    #     print("Batch of labels has shape: ", labels.shape)
