import torch
# 加噪的次数
timesteps_ = 500
path = "./dataset/dataset_GHI_2.csv"
step = 168  # one day data
batch_size = 32
epochs = 50000
log_batch = 20


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

