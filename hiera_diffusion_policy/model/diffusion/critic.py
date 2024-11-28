from typing import Union
import logging
import torch
import torch.nn as nn
import einops
import time
import copy
from hiera_diffusion_policy.model.diffusion.pointcloud_encoder import PointNetEncoder
from hiera_diffusion_policy.model.diffusion.mlps import StateEncoder

logger = logging.getLogger(__name__)



class Critic(nn.Module):
    def __init__(
            self,
            pcd_encoder: PointNetEncoder,
            state_dim,
            subgoal_dim,
            action_dim,
            mlp_dims=[512, 256, 128],
            ):
        super().__init__()

        # ************* condition *************
        self.pcd_encoder = pcd_encoder
        input_dim = state_dim + subgoal_dim + action_dim
        if pcd_encoder is not None:
            input_dim += self.pcd_encoder.out_dim

        last_dim = input_dim
        self.layers = nn.Sequential()
        for i, d in enumerate(mlp_dims):
            self.layers.append(nn.Linear(last_dim, d))
            self.layers.append(nn.ReLU())
            last_dim = d
        
        self.final_layer = nn.Linear(mlp_dims[-1], 1)
        

    def forward(self, pcd, state, subgoal, action):
        if self.pcd_encoder is not None:
            x = torch.concat(
                (self.pcd_encoder(pcd), state, subgoal, action), 
                dim=1)
        else:
            x = torch.concat((state, subgoal, action), dim=1)
            
        x = self.layers(x)
        x = self.final_layer(x)
        return x

    
class Critic2net(nn.Module):
    def __init__(
            self,
            pcd_encoder: PointNetEncoder,
            state_dim,
            subgoal_dim,
            action_dim,
            mlp_dims=[512, 256, 128],
            ):
        super().__init__()
        self.critic1 = Critic(
            pcd_encoder, state_dim, subgoal_dim, action_dim, mlp_dims)
        self.critic2 = Critic(
            copy.deepcopy(pcd_encoder), state_dim, subgoal_dim, action_dim, mlp_dims)

        logger.info("parameters number of Critic: %e", self.params_num())
    def forward(self, pcd, state, subgoal, action):
        q1 = self.critic1(pcd, state, subgoal, action)
        q2 = self.critic2(pcd, state, subgoal, action)
        return q1, q2

    def params_num(self):
        return sum(p.numel() for p in self.parameters())