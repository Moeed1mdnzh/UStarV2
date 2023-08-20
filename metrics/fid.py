import scipy
import torch
import numpy as np
from torchvision import transforms


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class FID_Score:
    def __init__(self, inceptionv3):
        inceptionv3.fc = Identity()
        self.inceptionv3 = inceptionv3
        self.preprocess = transforms.Compose([
            transforms.Resize(192),
            transforms.CenterCrop(192),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

    def _preprocess_images(self, images_1, images_2):
        images_1 = self.preprocess(images_1)
        images_2 = self.preprocess(images_2)
        return images_1, images_2

    def _predict_features(self, images_1, images_2):
        images_1, images_2 = self._preprocess_images(images_1, images_2)
        return self.inceptionv3(images_1), self.inceptionv3(images_2)

    def calculate_fid(self, images):
        images_1, images_2 = self._predict_features(images[0], images[1])
        images_1, images_2 = images_1.detach().cpu(
        ).numpy(), images_2.detach().cpu().numpy()
        mu1, sigma1 = images_1.mean(axis=0), np.cov(images_1, rowvar=False)
        mu2, sigma2 = images_2.mean(axis=0), np.cov(images_2, rowvar=False)

        ssdiff = np.sum((mu1 - mu2)**2.0)
        covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid
