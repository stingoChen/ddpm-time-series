# main
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

# [x1->x168];[169-2*168]
# [x1 168]; [x2 169]
# 数据加载 [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]->
"""
[[144]], [144]]; [x2, x3]; [x3, x4]
"""
dataloader = DataLoader(process_data.Train_dataset, batch_size=batch_size, shuffle=True)

model = models.diffusion(timesteps_).to(device)
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
# model = torch.load("./models/model50000")

# 训练循环
# epoch  表示 所有的数据训练次数
for epoch in range(epochs):
    loss_sum = 0
    # batch
    for batch, data in enumerate(dataloader):
        # t 表示 我们训练前向过程的时候 的 T
        # data.shape[0] 表示我们的batch 维度【batch=32，】
        # 需要初始化 32个 随机的前向过程 T
        # //2  先初始化16个 剩下16为了
        t = torch.randint(0, timesteps_, size=(int(data.shape[0]) // 2,))
        # 499，0；  200，299
        # cat tenor[499]  shape(1,1)  tenor[0]-> tensor[499, 0]
        t = torch.cat([t, timesteps_ - 1 - t], dim=0)
        t = t.unsqueeze(-1)

        # 加噪
        # datas, input_noise = utils.forward_process(data, t)
        # data [144 + 144]
        # cov_Data 是前一天的数据 datas 是后一天的数据
        cov_data, datas = data[:, :step], data[:, step:]
        # 我们只对 后一天的数据 进行加噪
        datas, input_noise = utils.forward_process(datas, t)
        # 放入gpu
        x = utils.make_cuda(datas.to(torch.float32))
        input_noise = utils.make_cuda(input_noise.to(torch.float32))
        t = utils.make_cuda(t)
        cov_x = utils.make_cuda(cov_data.to(torch.float32))

        # 训练
        optimizer.zero_grad()
        y = model(x, t, cov_x)
        loss_ = loss(y, input_noise)
        loss_.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()

        loss_sum += loss_
        if (batch + 1) % log_batch == 0:
            print("Epoch [%.3d/%.3d] Batch [%.3d/%.3d]: MSE_loss=%.4f"
                  % (epoch + 1,
                     epochs,
                     batch + 1,
                     len(dataloader),
                     loss_sum / log_batch,
                     )
                  )
            loss_sum = 0
    # ----------------------------save---------------------------
    # 因为loss 没有办法判断 是否训练完成，所以 每500轮进行采样
    if (epoch + 1) % 500 == 0:
        # 存模型
        torch.save(model, "./models/model%s_%s" % (epoch + 1, (loss_sum / log_batch).item()))
        # --------------------sampling-------------------------------
        dict_ = {}
        dataloader_test = DataLoader(process_data.Test_dataset, batch_size=1)
        for test_x_cov in dataloader_test:
            # 3 意思 初始化噪声-> 去噪500从 这个过程 重复三次
            # 如果三次结果 都是很乱的话，就说明 要么是模型坏了 要么就是还没训练完
            for ppp in range(3):
                # torch.cuda.empty_cache()
                # test_x_cov 前一天的数据
                # p_sample_loop 是反向过程函数
                x_seq = utils.p_sample_loop(model, [1, step], timesteps_, utils.betas, utils.one_minus_alphas_bar_sqrt,
                                            test_x_cov)
                # x_seq[-1] 去最后一列 因为去噪的最后一个结果是我们的最终结果
                pre_data = x_seq[-1].cpu().detach().numpy()[0]
                # print(pre_data)
                dict_[ppp] = pre_data
        df = pd.DataFrame(dict_)
        df.to_csv("./csv/%s.csv" % (epoch + 1))


dict_ = {}
dataloader_test = DataLoader(process_data.Test_dataset, batch_size=1)
for test_x_cov in dataloader_test:
    for ppp in range(20):
        x_seq = utils.p_sample_loop(model, [1, step], timesteps_, utils.betas, utils.one_minus_alphas_bar_sqrt, test_x_cov)
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


