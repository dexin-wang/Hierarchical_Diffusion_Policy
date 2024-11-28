import torch
import torch.nn as nn


class StateEncoder(nn.Module):
    def __init__(
            self, 
            state_dim,
            mlp_dims=[128, 256, 256]):
        super().__init__()

        last_dim = state_dim
        self.out_dim = mlp_dims[-1]
        self.mlps = nn.Sequential()
        for i, d in enumerate(mlp_dims):
            self.mlps.append(nn.Linear(last_dim, d))
            if i < len(mlp_dims)-1:
                self.mlps.append(nn.Mish())
            last_dim = d

    def forward(self, x: torch.Tensor):
        """
        x: (B, d)
        return: (B, d')
        """
        assert torch.is_tensor(x), 'x mush be torch.Tensor'
        return self.mlps(x)

    def params_num(self):
        return sum(p.numel() for p in self.parameters())
