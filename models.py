import torch.nn as nn
import torch
import math
from config import *
from utils import *


class ResnetBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = nn.Linear(step, 72)
        self.block2 = nn.Linear(72, step)
        self.relu = nn.ReLU()

    def forward(self, x):

        y = self.block1(x)
        y = self.block2(y)

        return self.relu(y) + x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=step, max_len=step):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # shape [5000] -> [5000, 1]

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(144.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = make_cuda(pe.unsqueeze(0))   # pe[1, 5000, 512]  [batch , max_len, number of feature]
        self.pe = pe

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class diffusion(nn.Module):
    def __init__(self, f_step):
        super(diffusion, self).__init__()
        self.step = step  # one day data
        self.f_step = f_step  # noising step
        self.dropout = 0.1    # Transformer dropout
        self.PE = PositionalEncoding()  # position encoding

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.lrelu = nn.LeakyReLU(0.2)
        # ---------------------mlp layer-----------------------
        self.mlp1 = nn.Linear(1, step)
        self.mlp2 = nn.Linear(1, step)
        # --------------------Transformer block-----------------------
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=step, nhead=8, dropout=self.dropout, batch_first=True)
        self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=step, nhead=8, dropout=self.dropout, batch_first=True)
        self.encoder_layer3 = nn.TransformerEncoderLayer(d_model=step, nhead=8, dropout=self.dropout, batch_first=True)

        self.encoder_layer4 = nn.TransformerEncoderLayer(d_model=step, nhead=8, dropout=self.dropout, batch_first=True)
        self.encoder_layer5 = nn.TransformerEncoderLayer(d_model=step, nhead=8, dropout=self.dropout, batch_first=True)
        self.encoder_layer6 = nn.TransformerEncoderLayer(d_model=step, nhead=8, dropout=self.dropout, batch_first=True)

        self.encoder_layer7 = nn.TransformerEncoderLayer(d_model=step, nhead=8, dropout=self.dropout, batch_first=True)
        self.encoder_layer8 = nn.TransformerEncoderLayer(d_model=step, nhead=8, dropout=self.dropout, batch_first=True)

        # ------------------------embedding layer-----------------------------
        self.embedding = nn.Embedding(f_step, step)
        self.embedding2 = nn.Embedding(f_step, step)
        self.embedding3 = nn.Embedding(f_step, step)

        # ---------------------------------------------------------
        self.resnet_x_min_block1 = ResnetBlock()  # [144, 144]
        self.resnet_x_min_block2 = ResnetBlock()
        self.resnet_x_min_block3 = ResnetBlock()
        # ---------------------------------------------------------

        self.fc5 = nn.Linear(step, 1)

    def forward(self, x, t, x_t_min_1):
        # x [batch, feature] -> [batch, feature, 1] for Transformer input
        x = x.unsqueeze(-1)
        x = self.mlp1(x)
        x_t_min_1 = x_t_min_1.unsqueeze(-1)
        x_t_min_1 = self.mlp1(x_t_min_1)

        # ------------------add position encoding--------------------------
        x = self.PE(x)

        # -------------------emb--------------------
        emb1 = self.embedding(t).squeeze(1).unsqueeze(-1)
        emb2 = self.embedding2(t).squeeze(1).unsqueeze(-1)
        emb3 = self.embedding3(t).squeeze(1).unsqueeze(-1)
        # -----------------------net-------------------------------
        x = x + emb1
        x = self.encoder_layer(x)
        x_t_min_1 = self.resnet_x_min_block1(x_t_min_1)
        x = self.encoder_layer2(x + x_t_min_1)
        x = self.relu(x)
        x = self.encoder_layer3(x)

        x = x + emb2
        x = self.encoder_layer4(x)
        x_t_min_1 = self.resnet_x_min_block2(x_t_min_1)
        x = self.encoder_layer5(x + x_t_min_1)
        x = self.relu(x)
        x = self.encoder_layer6(x)

        x = x + emb3
        x = self.encoder_layer7(x)
        x_t_min_1 = self.resnet_x_min_block3(x_t_min_1)
        x = self.encoder_layer8(x + x_t_min_1)
        x = self.relu(x)

        x = self.fc5(x)

        # x = self.PE(x)
        return x.squeeze(-1)



