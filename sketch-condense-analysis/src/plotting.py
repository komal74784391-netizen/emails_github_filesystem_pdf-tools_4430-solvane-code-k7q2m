import matplotlib.pyplot as plt
import pandas as pd


def plot_condensation_curve(csv_path, output_path):
    frame = pd.read_csv(csv_path)
    plt.figure(figsize=(5, 3.2))
    plt.plot(frame["ratio"], frame["accuracy"], marker="o")
    plt.xlabel("condensation ratio")
    plt.ylabel("accuracy")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
