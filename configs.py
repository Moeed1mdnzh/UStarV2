import torch

SHIFT_LIMIT = 0.4

BATCH_SIZE = 84

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DISC_LOSS = torch.nn.BCELoss()
GEN_LOSS_1 = torch.nn.BCELoss()
GEN_LOSS_2 = torch.nn.L1Loss()

N_EPOCHS = 40


