import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import einops


class SelfAttention(nn.Module):
    def __init__(self, input_dim, atten_dim):
        super(SelfAttention, self).__init__()
        self.query_layer = nn.Linear(input_dim, atten_dim)
        self.key_layer = nn.Linear(input_dim, atten_dim)
        self.value_layer = nn.Linear(input_dim, atten_dim)
    
    def forward(self, x):
        query = self.query_layer(x)
        key = self.key_layer(x)
        value = self.value_layer(x)
        
        # 计算注意力权重
        attention_weights = torch.matmul(query, key.transpose(1, 2))
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        # 对value加权求和
        attention_output = torch.matmul(attention_weights, value)
        return attention_output


class PointSAEncoder(nn.Module):
    def __init__(self, input_dim, atten_dim, output_dim):
        super(PointSAEncoder, self).__init__()
        self.attention = SelfAttention(input_dim, atten_dim)
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.output_layer = nn.Linear(atten_dim, output_dim)
        self.out_dim = output_dim
    
    def forward(self, x):
        # 输入点云数据的形状为 (B, N, C)
        # 对每个点进行自注意力编码
        attention_output = self.attention(x)
        
        # 对自注意力编码后的特征进行全局平均池化，将每个点的特征汇总成一维特征向量
        pooled_features = self.global_pooling(attention_output.permute(0, 2, 1)).squeeze(-1)
        
        # 输出一维特征向量
        output_features = self.output_layer(pooled_features)
        return output_features
    
    def params_num(self):
        return sum(p.numel() for p in self.parameters())
    

class PointNetEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 mlp_dims=[128, 256],
                 pool=True):
        super(PointNetEncoder, self).__init__()
        
        last_dim = input_dim
        self.convs = nn.Sequential()
        for i, d in enumerate(mlp_dims):
            self.convs.append(nn.Conv1d(last_dim, d, 1))
            if i < len(mlp_dims)-1:
                self.convs.append(nn.ReLU())
            last_dim = d

        self.pool = pool
        self.input_channel = input_dim
        self.out_dim = mlp_dims[-1]

    def forward(self, x: torch.Tensor):
        """
        x: (B, C, N)
        """
        x = einops.rearrange(x, 'b n c -> b c n')
        B, D, N = x.size()
        x = self.convs(x)
        if self.pool:
            x = torch.max(x, 2, keepdim=True)[0]    # pooling (B, 256, 1)
            x = x.view(B, -1)    # (B， 256)
        return x

    def params_num(self):
        return sum(p.numel() for p in self.parameters())
    

class PointNetEncoderSep(nn.Module):
    def __init__(self,
                 input_dim,
                 pcd_num=2,
                 mlp_dims=[128, 256]):
        super(PointNetEncoderSep, self).__init__()
        #! input_dim为每个点云的维度，不是合并后点云的总维度
        
        last_dim = input_dim
        self.convs = nn.Sequential()
        for i, d in enumerate(mlp_dims):
            self.convs.append(nn.Conv1d(last_dim, d, 1))
            if i < len(mlp_dims)-1:
                self.convs.append(nn.ReLU())
            last_dim = d

        self.input_channel = input_dim
        self.pcd_num = pcd_num
        self.out_dim = int(mlp_dims[-1]*pcd_num)

    def forward(self, x: torch.Tensor):
        """
        x: (B, N, C)
        """
        #!!! 点云分别编码，而不是合在一起编码
        x = einops.rearrange(x, 'b n c -> b c n')
        pcd_num = x.shape[1] // self.input_channel
        assert pcd_num == self.pcd_num
        ic = self.input_channel
        embds = list()
        for i in range(pcd_num):
            y = x[:, ic*i:ic*(i+1)]
            B, D, N = y.size()
            y = self.convs(y)
            y = torch.max(y, 2, keepdim=True)[0]    # pooling (B, 256, 1)
            y = y.view(B, -1)    # (B， 256)
            embds.append(y)
        output = torch.concat(tuple(embds), dim=-1)
        return output

    def params_num(self):
        return sum(p.numel() for p in self.parameters())
    

class PointNetEncoderSepLN(nn.Module):
    
    def __init__(self,
                 input_dim,
                 pcd_num=2,
                 mlp_dims=[128, 256]):
        super(PointNetEncoderSepLN, self).__init__()
        #! input_dim为每个点云的维度，不是合并后点云的总维度
        
        last_dim = input_dim
        self.convs = nn.Sequential()
        for i, d in enumerate(mlp_dims[:-1]):
            self.convs.append(nn.Linear(last_dim, d))
            if i < len(mlp_dims)-1:
                self.convs.append(nn.LayerNorm(d))
                self.convs.append(nn.ReLU())
            last_dim = d
        
        self.final_projection = nn.Sequential(
            nn.Linear(mlp_dims[-2], mlp_dims[-1]),
            nn.LayerNorm(mlp_dims[-1])
        )

        self.input_channel = input_dim
        self.pcd_num = pcd_num
        self.out_dim = int(mlp_dims[-1]*pcd_num)

    def forward(self, x: torch.Tensor):
        """
        x: (B, N, C)
        """
        #!!! 点云分别编码，而不是合在一起编码
        pcd_num = x.shape[-1] // self.input_channel
        assert pcd_num == self.pcd_num
        ic = self.input_channel
        embds = list()
        for i in range(pcd_num):
            y = x[..., ic*i:ic*(i+1)]
            y = self.convs(y)
            y = torch.max(y, 1)[0]
            y = self.final_projection(y)
            embds.append(y)
        output = torch.concat(tuple(embds), dim=-1)
        return output

    def params_num(self):
        return sum(p.numel() for p in self.parameters())
    
