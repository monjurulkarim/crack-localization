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


device = ("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
num_epochs = 10
num_classes = 2
learning_rate = 0.001
batch_size = 32
shuffle = True
pin_memory = True
num_workers = 1
data_path = 'data/'


def random_image(len_img_list,test_dir,list_test):
    test_images = []
    for i in range(9):
        ran = random.randint(1,len_img_list)
        validation_img_path = os.path.join(test_dir, list_test[ran])
        test_images.append(validation_img_path)
    img_list = [Image.open(img_path) for img_path in test_images]

    return img_list

def save_image(img_list,pred_probs):
    fig = plt.figure(figsize=(10, 7))
    for i, img in enumerate(img_list):
        fig.add_subplot(3, 3, i+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title("{:.0f}% Negative,\n {:.0f}% Positive".format(100*pred_probs[i,0],
                                                                100*pred_probs[i,1]))

    fig.subplots_adjust(hspace= 0.8)
    fig.savefig('temp.png')
    return
# plt.show()

def inference():
    val_dir = os.path.join(data_path, 'test')
    test_dir = os.path.join(val_dir, p.type)
    list_test = os.listdir(test_dir)

    len_img_list = len(list_test)
    model = CrackClassifier(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.load_state_dict(torch.load('snapshot/crack_model.pth'))

    img_list = random_image(len_img_list, test_dir, list_test)
    # print(labels)

    test_batches = torch.stack([transform(img).to(device)
                                    for img in img_list])
    pred_logits_tensor = model(test_batches)
    pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()

    save_image(img_list,pred_probs)
    print('image saved')
    print('===========')
    return pred_probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='Positive',
                        help='crack type for visualization.')
    p = parser.parse_args()
    inference()
