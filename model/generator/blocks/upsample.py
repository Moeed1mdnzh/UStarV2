import torch


class Upsample(torch.nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()
        self.conv_block_1 = torch.nn.ConvTranspose2d(
            1024, 1024, (4, 4), stride=(2, 2), padding=1)
        self.bn_1 = torch.nn.BatchNorm2d(1024)
        self.act_1 = torch.nn.CELU(0.2)

        self.conv_block_2 = torch.nn.ConvTranspose2d(
            2048, 512, (4, 4), stride=(2, 2), padding=1)
        self.bn_2 = torch.nn.BatchNorm2d(512)
        self.act_2 = torch.nn.CELU(0.2)

        self.conv_block_3 = torch.nn.ConvTranspose2d(
            1024, 256, (4, 4), stride=(2, 2), padding=1)
        self.bn_3 = torch.nn.BatchNorm2d(256)
        self.act_3 = torch.nn.CELU(0.2)

        self.conv_block_4 = torch.nn.ConvTranspose2d(
            512, 128, (4, 4), stride=(2, 2), padding=1)
        self.bn_4 = torch.nn.BatchNorm2d(128)
        self.act_4 = torch.nn.CELU(0.2)

        self.conv_block_5 = torch.nn.ConvTranspose2d(
            256, 3, (4, 4), stride=(2, 2), padding=1)
        self.act_5 = torch.nn.Tanh()

        self.conv_block_alt = torch.nn.Conv2d(
            1024, 1024, (4, 4), stride=(2, 2), padding=1)
        self.act_alt = torch.nn.CELU(0.2)

    def forward(self, x):
        b = self.conv_block_alt(x[3])
        b = self.act_alt(b)
        d4 = self.conv_block_1(b)
        d4 = self.bn_1(d4)
        d4 = torch.cat([d4, x[3]], dim=1)
        d4 = self.act_1(d4)

        d5 = self.conv_block_2(d4)
        d5 = self.bn_2(d5)
        d5 = torch.cat([d5, x[2]], dim=1)
        d5 = self.act_2(d5)

        d6 = self.conv_block_3(d5)
        d6 = self.bn_3(d6)
        d6 = torch.cat([d6, x[1]], dim=1)
        d6 = self.act_3(d6)

        d7 = self.conv_block_4(d6)
        d7 = self.bn_4(d7)
        d7 = torch.cat([d7, x[0]], dim=1)
        d7 = self.act_4(d7)
        z = self.conv_block_5(d7)
        z = self.act_5(z)
        return z
