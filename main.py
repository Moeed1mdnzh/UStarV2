import os
import cv2
import time
import torch
import numpy as np
from torch.types import Device
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
D_opt = torch.optim.Adam(D_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
G_opt = torch.optim.Adam(G_model.parameters(), lr=0.0002, betas=(0.5, 0.999))

def d_train(x, labels, generator, discriminator):
    discriminator.zero_grad()
    d_labels_real = torch.ones(BATCH_SIZE, 1, device=DEVICE)
    d_proba_real = discriminator(labels)
    d_loss_real = DISC_LOSS(d_proba_real, d_labels_real)
    g_output = generator(x)
    d_proba_fake = discriminator(g_output)
    d_labels_fake = torch.zeros(BATCH_SIZE, 1, device=DEVICE)
    d_loss_fake = DISC_LOSS(d_proba_fake, d_labels_fake)
    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    D_opt.step()
    return d_loss.data.item()

def g_train(x, labels, generator, discriminator):
    generator.zero_grad()
    g_labels_real = torch.ones(BATCH_SIZE, 1, device=DEVICE)
    g_output = generator(x)
    d_proba_fake = discriminator(g_output)
    g_loss1 = torch.unsqueeze(GEN_LOSS_1(d_proba_fake, g_labels_real), dim=0)
    g_loss2 = torch.unsqueeze(GEN_LOSS_2(d_proba_fake, g_labels_real), dim=0)
    g_loss = torch.concat((g_loss1, g_loss2), dim=0)
    g_loss = torch.multiply(g_loss, torch.tensor([1., 100.]).to(DEVICE))
    g_loss = torch.add(g_loss[0], g_loss[1])
    g_loss.backward()
    G_opt.step()
    return g_loss.data.item()

def create_samples(g_model, input_z):
    g_output = g_model(input_z)
    images = torch.reshape(g_output, (BATCH_SIZE, *(192, 192, 3)))
    return (images+1)/2.0


for epoch in range(1, N_EPOCHS):
    d_losses, g_losses = 0, 0
    print(f"Training {epoch}/{N_EPOCHS}... ")
    for X_batch, y_batch in dataset:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        d_loss = d_train(X_batch, y_batch, G_model, D_model)
        d_losses += d_loss
        g_losses += g_train(X_batch, y_batch, G_model, D_model)
    print(f"Epoch {epoch}/{N_EPOCHS}  g_loss {g_losses / len(dataset)}  d_loss {d_losses / len(dataset)}")
    samples = create_samples(G_model, X_batch)
    plt.imshow(samples[0].detach().cpu().numpy())
    plt.show()