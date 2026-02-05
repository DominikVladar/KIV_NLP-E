import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv


data_test = pd.read_csv("cv03/data/csfd-test.tsv", sep="\t", quoting=csv.QUOTE_NONE, header=None).to_numpy()[1:]
data_train = pd.read_csv("cv03/data/csfd-train.tsv", sep="\t", quoting=csv.QUOTE_NONE, header=None).to_numpy()[1:]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
datasets = [
    {"data": data_train, "name": "train"},
    {"data": data_test, "name": "test"},
    {"data": np.vstack((data_train, data_test)), "name": "train + test"}
]

for ax, dataset in zip(axes, datasets):
    scores = dataset["data"][:, 1].astype(np.float32)
    bins = np.arange(4) - 0.5
    ax.hist(scores, bins=bins, rwidth=0.8)
    ax.set_xticks([0, 1, 2])
    ax.set_title(f"Histogram ({dataset['name']})")
    ax.set_xlabel("Sentiment label")
    ax.set_ylabel("Number of sentence pairs")
    ax.grid()
    mean = scores.mean()
    std = scores.std()
    print(f"{dataset['name']} - mean: {mean:.3f}, std dev: {std:.3f}")

plt.tight_layout()
plt.savefig("cv03/histogram_subplots.svg")
plt.show()