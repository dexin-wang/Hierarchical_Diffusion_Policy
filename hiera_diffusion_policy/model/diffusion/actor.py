from typing import Union
import logging
import torch
import torch.nn as nn
import einops
import time
from einops.layers.torch import Rearrange

from hiera_diffusion_policy.model.diffusion.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)
from hiera_diffusion_policy.model.diffusion.positional_embedding import TimestepEncoder
from hiera_diffusion_policy.model.diffusion.pointcloud_encoder import PointNetEncoder
from hiera_diffusion_policy.model.diffusion.mlps import StateEncoder

logger = logging.getLogger(__name__)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False    # True
            ):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x) # (batch, out_channels, horizon)
        embed = self.cond_encoder(cond) # (batch, cond_channels, 1)
        if self.cond_predict_scale:
            # FiLM
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1)    # (batch, 2, out_channels, 1)
            scale = embed[:,0,...]  # (batch, out_channels, 1)
            bias = embed[:,1,...]   # (batch, out_channels, 1)
            out = scale * out + bias
        else:
            out = out + embed

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)   # 残差连接
        return out


class Actor(nn.Module):
    def __init__(
            self,
            diffusion_step_encoder: TimestepEncoder,
            pcd_encoder: PointNetEncoder,
            action_dim,
            state_dim,
            subgoal_dim,
            use_subgoal,
            use_pcd=True,
            down_dims=[256,512,1024],
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False    # true
            ):
        super().__init__()

        # ************* condition *************
        self.diffusion_step_encoder = diffusion_step_encoder
        self.pcd_encoder = pcd_encoder
        self.use_subgoal = use_subgoal
        self.use_pcd = use_pcd
        
        # Unet action noise predicter
        cond_dim = self.diffusion_step_encoder.out_dim + state_dim
        if use_pcd:
            cond_dim += self.pcd_encoder.out_dim
        if use_subgoal:
            cond_dim += subgoal_dim

        # ************* action noise predict *************
        all_dims = [action_dim] + list(down_dims)
        in_out = list(zip(all_dims[:-1], all_dims[1:])) # [(in_dim, 256), (256, 512), (512, 1024)]
        # * mid_modules
        mid_dim = all_dims[-1]  # 1024
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
        ])

        # * down_modules
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        # * up_modules
        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):  # (512, 1024), (256, 512)
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size=kernel_size),
            nn.Conv1d(down_dims[0], action_dim, 1),
        )

        logger.info("parameters number of Actor: %e", self.params_num())


    def forward(
            self, 
            pcd: torch.Tensor,
            state: torch.Tensor,
            subgoal: torch.Tensor=None,
            noised_actions: torch.Tensor=None,
            timestep: torch.Tensor=None
            ):
        """ """

        if not torch.is_tensor(timestep):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timestep = torch.tensor([timestep], dtype=torch.long, device=state.device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
            timestep = timestep[None].to(state.device)
        timestep = timestep.expand(state.shape[0])
        
        # ************** encode conditions **************
        timestep_emb = self.diffusion_step_encoder(timestep)    # (b, C1)
        
        cond = (timestep_emb, state)

        if self.use_pcd and pcd is not None:
            pcd_emb = self.pcd_encoder(pcd)
            cond += (pcd_emb,)
        if self.use_subgoal and subgoal is not None:
            cond += (subgoal,)
        
        global_feature = torch.concat(cond, dim=1)

        # ************** state/action **************
        x = einops.rearrange(noised_actions, 'b h c -> b c h')
        # down module
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)
        # mid module
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)
        # up module
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)  # (batch, in_channel, horizon)
        x = einops.rearrange(x, 'b c h -> b h c')   # (batch, horizon, in_channel) 和输入sample的shape一样
        
        return x


    def params_num(self):
        return sum(p.numel() for p in self.parameters())

