import torch

timesteps_ = 500
path = "./dataset/dataset_GHI_2.csv"
step = 144  # one day data
batch_size = 32
epochs = 50000
log_batch = 20


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

