import os
import cv2
import glob
import torch
import numpy as np
import torchvision
import os.path as osp
import RRDBNet_arch as arch
from model.weight_init import init_weights
from model.generator.generator import Generator


class Inference:
    def __init__(self, device="cpu"):
        self._device = device
        self._tfs = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                                                     std=(0.5, 0.5, 0.5))])
        self._g_model = Generator()
        self._g_model.apply(init_weights)
        self._g_opt = torch.optim.Adam(
            self._g_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self._esrgan = arch.RRDBNet(3, 3, 64, 23, gc=32)

    def _prep(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self._tfs(image)

    def _predict(self, image):
        pred = self._g_model(image.unsqueeze(0)).squeeze().permute(1, 2, 0)
        pred = (pred + 1) / 2.0
        pred = pred * 255.0
        pred = np.uint8(pred.detach().cpu().numpy())
        return pred

    def initialize(self):
        G_state = torch.load(os.sep.join(
            ["pre_trained_models", "generator_weights_7.pt"]), map_location=torch.device(self._device))
        self._g_model.load_state_dict(G_state["state_dict"])
        self._g_opt.load_state_dict(G_state["optimizer"])
        self._esrgan.load_state_dict(torch.load(os.sep.join(
            ["pre_trained_models", "RRDB_ESRGAN_x4.pth"])), strict=True)
        self._esrgan.eval()
        self._esrgan = self._esrgan

    def generate(self, image):
        prep = self._prep(image)
        pred = self._predict(prep)
        img = pred * 1.0 / 255
        img = torch.from_numpy(np.transpose(
            img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR
        with torch.no_grad():
            output = self._esrgan(img_LR).data.squeeze(
            ).float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        return output

# For test
if __name__ == "__main__":
    inference = Inference()
    inference.initialize()
    import time
    pre = time.time()
    image = cv2.imread(os.sep.join(["previews", "teststar2.jpg"]))
    image = inference.generate(image)
    print(time.time()-pre)
    cv2.imwrite("result.png", image)
    image = cv2.imread("result.png")
    cv2.imshow("", image)
    cv2.waitKey(0)
