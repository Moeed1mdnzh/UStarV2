import torch 

class MinibatchStddev(torch.nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean = torch.mean(x, dim=0, keepdims=True)
        diff = torch.square(x - mean)
        diff = torch.mean(diff, dim=0, keepdims=True) + self.eps
        stddev = torch.sqrt(diff) 
        stddev_mean = torch.mean(stddev)
        shape = x.shape
        y = torch.tile(stddev_mean, (shape[0], 1, shape[2], shape[3]))
        return torch.cat([x, y], 1) 