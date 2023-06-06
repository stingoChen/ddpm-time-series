import torch
import numpy as np
import math
import utils
import process_data
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
from config import *
import models
import pandas as pd


dataloader = DataLoader(process_data.Train_dataset, batch_size=batch_size, shuffle=True)

# model = models.diffusion(timesteps_).to(device)
# loss = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
model = torch.load("./models/model50000")

# for epoch in range(epochs):
#     loss_sum = 0
#     for batch, data in enumerate(dataloader):
#
#         t = torch.randint(0, timesteps_, size=(int(data.shape[0]) // 2,))
#         t = torch.cat([t, timesteps_ - 1 - t], dim=0)
#         t = t.unsqueeze(-1)
#
#         datas, input_noise = utils.forward_process(data, t)
#
#         cov_data, datas = data[:, :144], data[:, 144:]
#         datas, input_noise = utils.forward_process(datas, t)
#
#         x = utils.make_cuda(datas.to(torch.float32))
#         input_noise = utils.make_cuda(input_noise.to(torch.float32))
#         t = utils.make_cuda(t)
#         cov_x = utils.make_cuda(cov_data.to(torch.float32))
#
#         optimizer.zero_grad()
#         y = model(x, t, cov_x)
#         loss_ = loss(y, input_noise)
#         loss_.backward()
#         # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
#         optimizer.step()
#
#         loss_sum += loss_
#         if (batch + 1) % log_batch == 0:
#             print("Epoch [%.3d/%.3d] Batch [%.3d/%.3d]: MSE_loss=%.4f"
#                   % (epoch + 1,
#                      epochs,
#                      batch + 1,
#                      len(dataloader),
#                      loss_sum / log_batch,
#                      )
#                   )
#             loss_sum = 0
#     # ----------------------------save---------------------------
#     if (epoch + 1) % 500 == 0:
#         torch.save(model, "./models/model%s_%s" % (epoch + 1, (loss_sum / log_batch).item()))
#         # --------------------sampling-------------------------------
#         dict_ = {}
#         dataloader_test = DataLoader(process_data.Test_dataset, batch_size=1)
#         for test_x_cov in dataloader_test:
#             for ppp in range(3):
#                 # torch.cuda.empty_cache()
#                 x_seq = utils.p_sample_loop(model, [1, 144], timesteps_, utils.betas, utils.one_minus_alphas_bar_sqrt,
#                                             test_x_cov)
#                 pre_data = x_seq[-1].cpu().detach().numpy()[0]
#                 # print(pre_data)
#                 dict_[ppp] = pre_data
#         df = pd.DataFrame(dict_)
#         df.to_csv("./csv/%s.csv" % (epoch + 1))


dict_ = {}
dataloader_test = DataLoader(process_data.Test_dataset, batch_size=1)
for test_x_cov in dataloader_test:
    for ppp in range(20):
        x_seq = utils.p_sample_loop(model, [1, 144], timesteps_, utils.betas, utils.one_minus_alphas_bar_sqrt, test_x_cov)
        pre_data = x_seq[-1].cpu().detach().numpy()[0]
        dict_[ppp] = pre_data
        # lllist = {}
        # for n, ooo in enumerate(x_seq):
        #     lllist[n] = ooo.cpu().detach().numpy()[0]
        # df3 = pd.DataFrame(lllist)
        # df3.to_csv("./csv/%s.csv" % ppp)

df = pd.DataFrame(dict_)
df.to_csv("result.csv")

# df1 = pd.DataFrame(process_data.Test_dataset_1)
# df1.to_csv("result2.csv")


