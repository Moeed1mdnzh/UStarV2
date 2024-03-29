import torch
import progressbar

SHIFT_LIMIT = 0.4

BATCH_SIZE = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DISC_LOSS = torch.nn.BCELoss()
GEN_LOSS_1 = torch.nn.BCELoss()
GEN_LOSS_2 = torch.nn.L1Loss()

N_EPOCHS = 5

widgets = [progressbar.Percentage(), " ", progressbar.GranularBar(left='', right='|'),
           " ", progressbar.ETA(), " ", progressbar.Variable("g_loss"), " ",
           progressbar.Variable("d_loss")]

widgets_2 = [progressbar.Percentage(), " ", progressbar.GranularBar(left='', right='|'),
             " ", progressbar.ETA(), " ", progressbar.Variable("ImageNumber")]

DATA_NAME = "data.jpg"

DATA_CONTROL = 25

FID = True
