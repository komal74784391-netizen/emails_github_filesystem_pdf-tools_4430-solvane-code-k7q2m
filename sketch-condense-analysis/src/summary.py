import pandas as pd


def summarise(csv_path):
    frame = pd.read_csv(csv_path)
    return frame.groupby("dataset")["accuracy"].agg(["mean", "std"])
