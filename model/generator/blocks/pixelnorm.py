import torch 

class PixelNorm(torch.nn.Module):
    def __init__(self, eps=1.0e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        vals = torch.square(x)
        mean = torch.mean(vals, axis=1, keepdims=True)
        mean += self.eps
        l2 = torch.sqrt(mean)
        return x / l2 