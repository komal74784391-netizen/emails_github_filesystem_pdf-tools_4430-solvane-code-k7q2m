import pandas as pd
import torch
from torch.utils.data import Dataset


class CrystalDataset(Dataset):
    def __init__(self, csv_path, max_atoms=64, atom_dim=128):
        self.frame = pd.read_csv(csv_path)
        self.max_atoms = max_atoms
        self.atom_dim = atom_dim

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        features = torch.zeros(self.max_atoms, self.atom_dim)
        n = min(int(row["n_atoms"]), self.max_atoms)
        features[:n] = torch.randn(n, self.atom_dim)
        return features, torch.tensor(float(row["target"]), dtype=torch.float32)
