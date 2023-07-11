import torch

SHIFT_LIMIT = 0.4

BATCH_SIZE = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOSS_FN = torch.nn.BCELoss()


