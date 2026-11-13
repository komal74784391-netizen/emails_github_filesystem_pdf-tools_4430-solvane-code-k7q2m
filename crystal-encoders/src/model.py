import torch
import torch.nn as nn


class CrystalEncoder(nn.Module):
    def __init__(self, atom_dim=128, hidden=256):
        super().__init__()
        self.atom_embed = nn.Linear(atom_dim, hidden)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden, nhead=8, batch_first=True), num_layers=4
        )

    def forward(self, atom_features, mask=None):
        x = self.atom_embed(atom_features)
        return self.encoder(x, src_key_padding_mask=mask)


class ContrastiveCrystalEncoder(nn.Module):
    def __init__(self, atom_dim=128, hidden=256, projection=64):
        super().__init__()
        self.encoder = CrystalEncoder(atom_dim, hidden)
        self.projection = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, projection)
        )

    def forward(self, atom_features, mask=None):
        pooled = self.encoder(atom_features, mask).mean(dim=1)
        return nn.functional.normalize(self.projection(pooled), dim=-1)
