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

G_model = Generator().to(DEVICE)
D_model = Discriminator().to(DEVICE)
G_model.apply(init_weights)
D_model.apply(init_weights)

def d_train(x, discriminator):
    discriminator.zero_grad()
    batch_size = x.size(0)
    d_labels_real = torch.ones(batch_size, 1, DEVICE=DEVICE)
    d_proba_real = discriminator(x)
    d_loss_real = loss_fn(d_proba_real, d_labels_real)
    g_output = generator(x)
    d_proba_fake = discriminator(g_output)
    d_labels_fake = torch.zeros(batch_size, 1, DEVICE=DEVICE)
    d_loss_fake = loss_fn(d_proba_fake, d_labels_fake)
    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    d_optimizer.step()
    return d_loss.data.item()

def g_train(x, generator):
    generator.zero_grad()
    batch_size = x.size(0)
    g_labels_real = torch.ones(batch_size, 1, DEVICE=DEVICE)
    g_output = generator(x)
    d_proba_fake = discriminator(g_output)
    g_loss = loss_fn(d_proba_fake, g_labels_real)
    g_loss.backward()
    g_optimizer.step()
    return g_loss.data.item()


    