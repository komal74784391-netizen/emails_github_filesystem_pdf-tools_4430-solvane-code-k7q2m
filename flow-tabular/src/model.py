import torch
import torch.nn as nn


class TabularVelocityField(nn.Module):
    def __init__(self, n_features, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features * 2 + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_features),
        )

    def forward(self, x, t, cond=None):
        inputs = [x, torch.full_like(x, t)] if cond is None else [x, torch.full_like(x, t), cond]
        return self.net(torch.cat(inputs, dim=-1))
