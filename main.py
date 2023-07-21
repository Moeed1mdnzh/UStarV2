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

dataset = ImageDataset(im_paths, label_paths, tfs)
dataset = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

G_model = Generator().to(DEVICE)
D_model = Discriminator().to(DEVICE)
G_model.apply(init_weights)
D_model.apply(init_weights)
D_opt = torch.optim.Adam(D_model.parameters(), lr=0.002, betas=(0.5, 0.999))
G_opt = torch.optim.Adam(G_model.parameters(), lr=0.002, betas=(0.5, 0.999))

def d_train(x, labels, generator, discriminator):
    discriminator.zero_grad()
    d_labels_real = torch.ones(x.size(0), 1, device=DEVICE) - 0.1
    d_proba_real = discriminator(labels)
    d_loss_real = DISC_LOSS(d_proba_real, d_labels_real)
    g_output = generator(x)
    d_proba_fake = discriminator(g_output)
    d_labels_fake = torch.zeros(x.size(0), 1, device=DEVICE)
    d_loss_fake = DISC_LOSS(d_proba_fake, d_labels_fake)
    d_loss = d_loss_real + d_loss_fake
    d_loss = d_loss * 0.5
    d_loss.backward()
    D_opt.step()
    return d_loss.data.item()


def g_train(x, labels, generator, discriminator):
    generator.zero_grad()
    g_labels_real = torch.ones(x.size(0), 1, device=DEVICE) - 0.1
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

def quick_inference(g_model, input_z):
    g_output = g_model(input_z.unsqueeze(dim=0))
    image = g_output.squeeze().permute(1, 2, 0)
    input_z = input_z.permute(1, 2, 0)
    return (image+1)/2.0, (input_z+1)/2.0

for epoch in range(1, N_EPOCHS):
    d_losses, g_losses = 0, 0
    print(f"Training {epoch}/{N_EPOCHS}")
    pbar = progressbar.ProgressBar(max_value=len(dataset), widgets=widgets)
    for i, (X_batch, y_batch) in enumerate(dataset):
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        d_loss = d_train(X_batch, y_batch, G_model, D_model)
        d_losses += d_loss
        g_loss = g_train(X_batch, y_batch, G_model, D_model)
        g_losses += g_loss
        pbar.update(i, g_loss=g_loss, d_loss=d_loss)
    print(f"Epoch {epoch}/{N_EPOCHS}  g_loss {g_losses / len(dataset)}  d_loss {d_losses / len(dataset)}")
    pbar.finish()
    if epoch == 1:
        sample_image = X_batch[0]
        sample_label = y_batch[0]
    samples = quick_inference(G_model, sample_image)
    fig = plt.figure(figsize=(1, 3))
    plt.subplot(3, 1, 1)
    plt.imshow(samples[1].detach().cpu().numpy())
    plt.axis("off")
    plt.title("input")
    
    plt.subplot(3, 1, 2)
    plt.imshow(sample_label.detach().cpu().numpy())
    plt.axis("off")
    plt.title("target")
    
    plt.subplot(3, 1, 3)
    plt.imshow(samples[0].detach().cpu().numpy())
    plt.axis("off")
    plt.title("pred")
    
    plt.savefig("prediction.png", dpi=fig.dpi)
    