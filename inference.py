import cv2 
import torch
import numpy as np 
import torchvision
from model.weight_init import init_weights
from model.generator.generator import Generator

tfs = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

G_model = Generator()
G_model.apply(init_weights)
G_opt = torch.optim.Adam(G_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
G_state = torch.load("generator_weights_7.pt", map_location=torch.device('cpu'))
G_model.load_state_dict(G_state["state_dict"])
G_opt.load_state_dict(G_state["optimizer"])

image = cv2.imread("teststar4.jpg")
sample = tfs(image)
pred = G_model(sample.unsqueeze(0)).squeeze().permute(1, 2, 0)
pred = (pred+1)/2.0
pred = pred * 255.0
pred = np.uint8(pred.detach().cpu().numpy())
cv2.imwrite("testres.jpg", pred)
