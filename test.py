import os
import cv2
import time
import torch
import progressbar
import numpy as np
import torchvision
from configs import *
from imutils import paths
from datetime import datetime
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
print(im_paths)
print("""
      
      
      
      """)
print(label_paths)

dataset = ImageDataset(im_paths, label_paths, tfs)
dataset = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# for X_batch, y_batch in dataset:
#     break

# sample = (X_batch[32]+1)/2.0
# label = (y_batch[32]+1)/2.0
# sample = sample.permute(1, 2, 0).detach().cpu().numpy()
# label = label.permute(1, 2, 0).detach().cpu().numpy()
# cv2.imshow("", sample)
# cv2.imshow("label", label)
# cv2.waitKey(0)
####
####
#### Take samples outside of the loop
#### Stop the loop from experiencing final batch
####
for epoch in range(1, N_EPOCHS):
    d_losses, g_losses = 0, 0
    print(f"Training {epoch}/{N_EPOCHS}")
    for i, (X_batch, y_batch) in enumerate(dataset):
        break
    print(f"\nEpoch {epoch}/{N_EPOCHS}  g_loss {g_losses / len(dataset)}  d_loss {d_losses / len(dataset)}")
    if epoch == 1:
        sample_image = X_batch[0].permute(1, 2, 0)
        sample_label = y_batch[0]
        sample_label = sample_label.permute(1, 2, 0)
        sample_label = (sample_label+1)/2.0
        sample_image = (sample_image+1)/2.0
    fig = plt.figure(figsize=(3, 1))
    plt.subplot(1, 3, 1)
    plt.imshow(sample_image.detach().cpu().numpy())
    plt.axis("off")
    plt.title("input", fontsize=10, pad="2.0")
    
    plt.subplot(1, 3, 2)
    plt.imshow(sample_label.detach().cpu().numpy())
    plt.axis("off")
    plt.title("target", fontsize=10, pad="2.0")
    
    plt.savefig(f"prediction_{epoch}.png", dpi=200)
    print(fig.dpi)
    break