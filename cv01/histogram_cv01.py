from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    dataset1 = datasets.MNIST('data', train=True, download=True)
    dataset2 = datasets.MNIST('data', train=False)

    labels1 = dataset1.targets.numpy()
    labels2 = dataset2.targets.numpy()

    plt.figure(figsize=(18, 10))  # Nastaví velikost okna a PDF

    plt.subplot(3, 1, 1)
    counts1, bins1, patches1 = plt.hist(labels1, bins=np.arange(11)-0.5, label='Training set', rwidth=0.9)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(np.arange(10))
    plt.legend()
    plt.ylim(0, counts1.max() * 1.25)
    for i in range(len(counts1)):
        plt.text(bins1[i]+0.5, counts1[i], str(int(counts1[i])), ha='center', va='bottom', fontsize=14)

    plt.tight_layout(pad=2.0)

    plt.subplot(3, 1, 2)
    counts2, bins2, patches2 = plt.hist(labels2, bins=np.arange(11)-0.5, label='Testing set', rwidth=0.9)
    plt.xticks(np.arange(10))
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.legend()
    plt.ylim(0, counts2.max() * 1.25)
    for i in range(len(counts2)):
        plt.text(bins2[i]+0.5, counts2[i], str(int(counts2[i])), ha='center', va='bottom', fontsize=14)

    plt.tight_layout(pad=2.0)

    plt.subplot(3, 1, 3)
    mixed_counts = counts1 + counts2
    bins = np.arange(11)-0.5
    plt.hist(np.repeat(np.arange(10), mixed_counts.astype(int)), bins=bins, label='Combined set', rwidth=0.9)
    plt.xticks(np.arange(10))
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.legend()
    plt.ylim(0, mixed_counts.max() * 1.25)
    for i in range(len(mixed_counts)):
        plt.text(bins[i]+0.5, mixed_counts[i], str(int(mixed_counts[i])), ha='center', va='bottom', fontsize=14)

    plt.tight_layout(pad=2.0)     

    plt.savefig('cv01/histogram_cv01.png')
    plt.show()

