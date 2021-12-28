import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
from src.model import CrackClassifier
from src.dataloader import CrackDataset
import argparse
import cv2

#for grad-cam
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image



device = ("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

num_classes = 2
learning_rate = 0.001
batch_size = 32
data_path = 'data/'

methods = \
    {"gradcam": GradCAM,
     "scorecam": ScoreCAM,
     "gradcam++": GradCAMPlusPlus,
     "ablationcam": AblationCAM,
     "xgradcam": XGradCAM,
     "eigencam": EigenCAM,
     "eigengradcam": EigenGradCAM,
     "layercam": LayerCAM,
     "fullgrad": FullGrad}

use_cuda = torch.cuda.is_available()

def random_image(len_img_list,test_dir,list_test):
    test_images = []
    for i in range(9):
        ran = random.randint(1,len_img_list-1)
        validation_img_path = os.path.join(test_dir, list_test[ran])
        test_images.append(validation_img_path)

    return test_images


def xai():
    model = CrackClassifier(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.load_state_dict(torch.load('snapshot/crack_model.pth'))

    target_layers = [model.resnet.layer4[-1]]
    val_dir = os.path.join(data_path, 'test')
    test_dir = os.path.join(val_dir, p.type)
    list_test = os.listdir(test_dir)
    len_img_list = len(list_test)
    test_images = random_image(len_img_list, test_dir, list_test)

    if p.type == 'Negative':
        target_category = 0
    else:
        target_category = 1

    cam_algorithm = methods[p.method]
    cam = cam_algorithm(model=model, target_layers=target_layers, use_cuda=use_cuda)

    fig = plt.figure(figsize=(10, 7))
    for i, img in enumerate(test_images):
        image_path = img
        rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img,
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        gb_model = GuidedBackpropReLUModel(model=model, use_cuda=use_cuda)
        gb = gb_model(input_tensor, target_category=target_category)
        cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        cam_gb = deprocess_image(cam_mask * gb)
        gb = deprocess_image(cam_image)
        file_name = img.split('/')[-1]
        #Draw figure
        fig.add_subplot(3, 3, i+1)
        plt.imshow(cam_image)
        plt.axis('off')
        plt.title(file_name)
    fig.subplots_adjust(hspace= 0.8)
    fig.savefig(p.method +'.png')
    print('-----Finished------')
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='Positive',
                        help='crack type for visualization.')
    parser.add_argument('--method', type=str, default='gradcam',
                        help='crack type for visualization.')
    p = parser.parse_args()
    xai()
