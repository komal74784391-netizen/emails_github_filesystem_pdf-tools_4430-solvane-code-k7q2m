import torch
from .model import TabularVelocityField


class FlowMatcher:
    def __init__(self, n_features, sigma=0.05):
        self.field = TabularVelocityField(n_features)
        self.sigma = sigma

    def loss(self, x1):
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], 1)
        xt = (1 - t) * x0 + t * x1
        target = x1 - x0
        return torch.nn.functional.mse_loss(self.field(xt, t), target)

    def sample(self, n, dim):
        x = torch.randn(n, dim)
        for step in range(50):
            t = step / 50
            x = x + self.field(x, t) * (1 / 50)
        return x
