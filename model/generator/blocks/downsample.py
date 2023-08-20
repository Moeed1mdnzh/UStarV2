import torch


class Downsample(torch.nn.Module):
    def __init__(self):
        super(Downsample, self).__init__()
        self.conv_block_1 = torch.nn.Conv2d(
            3, 128, (4, 4), stride=(2, 2), padding=1)
        self.act_1 = torch.nn.CELU(0.2)

        self.conv_block_2 = torch.nn.Conv2d(
            128, 256, (4, 4), stride=(2, 2), padding=1)
        self.bn_2 = torch.nn.BatchNorm2d(256)
        self.act_2 = torch.nn.CELU(0.2)

        self.conv_block_3 = torch.nn.Conv2d(
            256, 512, (4, 4), stride=(2, 2), padding=1)
        self.bn_3 = torch.nn.BatchNorm2d(512)
        self.act_3 = torch.nn.CELU(0.2)

        self.conv_block_4 = torch.nn.Conv2d(
            512, 1024, (4, 4), stride=(2, 2), padding=1)
        self.bn_4 = torch.nn.BatchNorm2d(1024)
        self.act_4 = torch.nn.CELU(0.2)

    def forward(self, x):
        e1 = self.conv_block_1(x)
        e1 = self.act_1(e1)

        e2 = self.conv_block_2(e1)
        e2 = self.bn_2(e2)
        e2 = self.act_2(e2)

        e3 = self.conv_block_3(e2)
        e3 = self.bn_3(e3)
        e3 = self.act_3(e3)

        e4 = self.conv_block_4(e3)
        e4 = self.bn_4(e4)
        e4 = self.act_4(e4)
        return e1, e2, e3, e4
