import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv


data_test = pd.read_csv("cv02/data/anlp01-sts-free-test.tsv", sep="\t", quoting=csv.QUOTE_NONE, header=None).to_numpy()
data_train = pd.read_csv("cv02/data/anlp01-sts-free-train.tsv", sep="\t", quoting=csv.QUOTE_NONE, header=None).to_numpy()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
datasets = [
    {"data": data_train, "name": "train"},
    {"data": data_test, "name": "test"},
    {"data": np.vstack((data_train, data_test)), "name": "train + test"}
]

for ax, dataset in zip(axes, datasets):
    scores = dataset["data"][:, 2].astype(np.float32)
    ax.hist(scores, bins=16)
    ax.set_title(f"Histogram ({dataset['name']})")
    ax.set_xlabel("Similarity score")
    ax.set_ylabel("Number of sentence pairs")
    ax.grid()
    mean = scores.mean()
    std = scores.std()
    print(f"{dataset['name']} - mean: {mean:.3f}, std dev: {std:.3f}")

plt.tight_layout()
plt.savefig("cv02/histogram_subplots.svg")
plt.show()