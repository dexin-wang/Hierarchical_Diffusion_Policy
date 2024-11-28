import math
import torch
import torch.nn as nn
from typing import Union

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        x: (batch,)
        return: (batch, dim)
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]     # (batch, 1) * (1, half_dim)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1) # (batch, dim)
        
        return emb

"""
和transformer中的position embeding实现有点差异，原始实现参考：https://zhuanlan.zhihu.com/p/360539748
"""

class TimestepEncoder(nn.Module):
    def __init__(self, diffusion_step_embed_dim=256):
        super().__init__()

        d = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(d),
            nn.Linear(d, d * 4),
            nn.Mish(),
            nn.Linear(d * 4, d),
        )
        self.out_dim = diffusion_step_embed_dim

    def forward(self, timestep: torch.Tensor):
        """
        timestep: (B,) or int, diffusion step
        output: (B, d)
        """
        return self.diffusion_step_encoder(timestep) # sin-cos position emb (batch, 256)

    def params_num(self):
        return sum(p.numel() for p in self.parameters())