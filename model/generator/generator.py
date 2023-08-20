import torch
from model.generator.blocks.upsample import Upsample
from model.generator.blocks.downsample import Downsample


class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.downsample = Downsample()
        self.upsample = Upsample()

    def forward(self, x):
        downsampled = self.downsample(x)
        upsampled = self.upsample(downsampled)
        return upsampled
