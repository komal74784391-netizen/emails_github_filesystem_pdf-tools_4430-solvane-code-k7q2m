import torch
from torch.utils.data import DataLoader

from .data import CrystalDataset
from .model import ContrastiveCrystalEncoder


def main():
    dataset = CrystalDataset("data/mp20_train.csv")
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    model = ContrastiveCrystalEncoder()
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for epoch in range(80):
        for features, target in loader:
            loss = torch.nn.functional.mse_loss(model(features).mean(-1), target)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    torch.save(model.state_dict(), "checkpoints/crystal-encoder.pt")


if __name__ == "__main__":
    main()
