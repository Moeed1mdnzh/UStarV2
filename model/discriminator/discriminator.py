import torch 

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_1 = torch.nn.Conv2d(3, 64, (4, 4), stride=(2, 2), padding=1)
        self.act_1 = torch.nn.LeakyReLU(0.2)
        
        self.conv_2 = torch.nn.Conv2d(64, 128, (4, 4), stride=(2, 2), padding=1)
        self.bn_2 = torch.nn.BatchNorm2d(128)
        self.act_2 = torch.nn.LeakyReLU(0.2)
        
        self.conv_3 = torch.nn.Conv2d(128, 256, (4, 4), stride=(2, 2), padding=1)
        self.bn_3 = torch.nn.BatchNorm2d(256)
        self.act_3 = torch.nn.LeakyReLU(0.2)
        
        self.conv_4 = torch.nn.Conv2d(256, 512, (4, 4), stride=(2, 2), padding=1)
        self.bn_4 = torch.nn.BatchNorm2d(512)
        self.act_4 = torch.nn.LeakyReLU(0.2)
        
        self.conv_5 = torch.nn.Conv2d(512, 1024, (4, 4), stride=(2, 2), padding=1)
        self.bn_5 = torch.nn.BatchNorm2d(1024)
        self.act_5 = torch.nn.LeakyReLU(0.2)
        
        self.flatten = torch.nn.Flatten()
        self.linear_0 = torch.nn.Linear(1024*6*6, 1)
        self.act_0 = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv_1(x)
        x = self.act_1(x)
        
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.act_2(x)
        
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.act_3(x)
        
        x = self.conv_4(x)
        x = self.bn_4(x)
        x = self.act_4(x)
        
        x = self.conv_5(x)
        x = self.bn_5(x)
        x = self.act_5(x)
        
        x = self.flatten(x)
        x = self.linear_0(x)
        x = self.act_0(x)
        return x 