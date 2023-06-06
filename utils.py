import torch
import math
from config import *


# ---------------------make cuda---------------------------
def make_cuda(Tensor_):
    if torch.cuda.is_available():
        return Tensor_.to("cuda")
    else:
        return Tensor_


# ------------------------betas schedule-------------------------
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    # return betas
    return torch.clamp(betas, 0, 0.999)


# ---------------------------forward process---------------------------
def forward_process(x_0, t):
    """
    it's the forward process.
    formula: q(x_t|x_{t-1}) = N[(1-beta) * x_{t-1}, beta]
    """
    if not torch.is_tensor(x_0):
        x_0 = torch.tensor(x_0)

    noise = torch.randn_like(x_0)  # [batch, step]
    n_max = torch.max(noise, dim=1).values.unsqueeze(-1)  # [1, batch]
    n_min = torch.min(noise, dim=1).values.unsqueeze(-1)
    noise = (noise - n_min) / (n_max - n_min) * 2 - 1

    alphas_t = alphas_bar_sqrt[t]
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
    # 在x[0]的基础上添加噪声
    return (alphas_t * x_0 + alphas_1_m_t * noise), noise


# ---------------------------- sample -----------------------------
def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt, x_cov_):
    """
    从x[T]采样t时刻的重构值
    """
    t = torch.tensor([t])

    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]

    x = make_cuda(x.to(torch.float32))
    t = make_cuda(t)
    x_cov_ = make_cuda(x_cov_.to(torch.float32))

    with torch.no_grad():
        eps_theta = model(x, t, x_cov_ )

    x = x.cpu().double()
    eps_theta = eps_theta.cpu().double()

    mean = (1 / (1. - betas[t]).sqrt()) * (x - (coeff * eps_theta))

    if t > 1:
        z = torch.randn_like(x)
        z_max = torch.max(z, dim=1).values.unsqueeze(-1)
        z_min = torch.min(z, dim=1).values.unsqueeze(-1)
        z = (z - z_min) / (z_max - z_min) * 2 - 1
    else:
        z = 0

    sigma_t = betas[t].sqrt()
    sample = mean + sigma_t * z

    return sample


def p_sample_loop(model, shape, n_steps, betas, one_minus_alphas_bar_sqrt, x_cov_):
    """从x[T]恢复x[T-1]、x[T-2]|...x[0]"""
    cur_x = torch.randn(shape).to(device)

    cur_x_max = torch.max(cur_x, dim=1).values.unsqueeze(-1)
    cur_x_min = torch.min(cur_x, dim=1).values.unsqueeze(-1)
    cur_x = (cur_x - cur_x_min) / (cur_x_max - cur_x_min) * 2 - 1

    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt, x_cov_)
        x_seq.append(cur_x)
    return x_seq


# region-------------------calc alpha etc-------------------------------------
betas = cosine_beta_schedule(timesteps_)
# 计算alpha、alpha_prod、alpha_prod_previous、alpha_bar_sqrt等变量的值
alphas = 1. - betas
alphas_prod = torch.cumprod(alphas, 0)
alphas_prod_p = torch.cat([torch.tensor([1.]), alphas_prod[:-1].float()], 0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1. - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1. - alphas_prod)
# endregion----------------------------------------
