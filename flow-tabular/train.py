import pandas as pd
import torch

from src.flow import FlowMatcher


def main():
    frame = pd.read_csv("data/adult.csv").select_dtypes("number")
    x = torch.tensor(frame.values, dtype=torch.float32)
    matcher = FlowMatcher(x.shape[1])
    optimiser = torch.optim.AdamW(matcher.field.parameters(), lr=1e-3)
    for epoch in range(200):
        loss = matcher.loss(x)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    torch.save(matcher.field.state_dict(), "checkpoints/flow.pt")


if __name__ == "__main__":
    main()
