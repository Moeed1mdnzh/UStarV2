import os
import cv2
import time
import torch
import numpy as np
import torchvision
from configs import *
from imutils import paths
import matplotlib.pyplot as plt
from model.weight_init import init_weights
from model.generator.generator import Generator
from model.discriminator.discriminator import Discriminator

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
tfs = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, im_paths, label_paths, transform):
        self.im_paths = im_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, index):
        image = cv2.imread(self.im_paths[index])
        label = cv2.imread(self.label_paths[index])
        image = self.transform(image)
        label = self.transform(label)
        return image, label

im_paths = list(paths.list_images(os.sep.join(["dataset", "images"])))
label_paths = list(paths.list_images(os.sep.join(["dataset", "labels"])))

dataset = ImageDataset(im_paths, label_paths, tfs)
dataset = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

G_model = Generator()
D_model = Discriminator()
G_model.apply(init_weights)
D_model.apply(init_weights)
print(G_model, D_model)

    