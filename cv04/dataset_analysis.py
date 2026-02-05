import numpy as np
import matplotlib.pyplot as plt
import transformers


def load_file(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as fp:
        seq = {"word": [], "label": []}
        for line in fp:
            if line == "\n":
                if seq["word"]:
                    data.append(seq)
                    seq = {"word": [], "label": []}
            else:
                split = line.strip().split(" ")
                if len(split) == 1:
                    split = line.strip().split("\t")
                word = " ".join(split[:-1])
                seq["word"].append(word)
                label = split[-1]
                seq["label"].append(label)
        if seq["word"]:
            data.append(seq)
    return data


def avg_len(data):
    return np.mean([len(seq["word"]) for seq in data])


def main():
    CNEC_data = [load_file(file_path) for file_path in ["cv04/data/train.txt", "cv04/data/dev.txt", "cv04/data/test.txt"]]
    UD_data = [load_file(file_path) for file_path in ["cv04/data-mt/train.txt", "cv04/data-mt/dev.txt", "cv04/data-mt/test.txt"]]


    print("CNEC dataset:")
    for i, data in enumerate(CNEC_data):
        print(f"\t{['train', 'dev', 'test'][i]}: {len(data)} sentences")
    print(f"\ttrain+dev+test: {sum(len(data) for data in CNEC_data)} sentences")

    print("UD dataset:")
    for i, data in enumerate(UD_data):
        print(f"\t{['train', 'dev', 'test'][i]}: {len(data)} sentences")
    print(f"\ttrain+dev+test: {sum(len(data) for data in UD_data)} sentences")


    print("CNEC dataset:")
    for i, data in enumerate(CNEC_data):
        print(f"\t{['train', 'dev', 'test'][i]}: {avg_len(data)} words")
    print(f"\ttrain+dev+test: {avg_len([seq for data in CNEC_data for seq in data])} words")

    print("UD dataset:")
    for i, data in enumerate(UD_data):
        print(f"\t{['train', 'dev', 'test'][i]}: {avg_len(data)} words")
    print(f"\ttrain+dev+test: {avg_len([seq for data in UD_data for seq in data])} words")


    tokenizer = transformers.BertTokenizerFast.from_pretrained("UWB-AIR/Czert-B-base-cased")
    CNEC_tokenized = []
    for data in CNEC_data:
        tokenized = []
        for seq in data:
            sentence = " ".join(seq["word"])
            tokenized.append(tokenizer.tokenize(sentence))
        CNEC_tokenized.append(tokenized)

    UD_tokenized = []
    for data in UD_data:
        tokenized = []
        for seq in data:
            sentence = " ".join(seq["word"])
            tokenized.append(tokenizer.tokenize(sentence))
        UD_tokenized.append(tokenized)

    print("CNEC dataset:")
    for i, data in enumerate(CNEC_tokenized):
        print(f"\t{['train', 'dev', 'test'][i]}: {np.mean([len(seq) for seq in data])} tokens")
    print(f"\ttrain+dev+test: {np.mean([len(seq) for data in CNEC_tokenized for seq in data])} tokens")

    print("UD dataset:")
    for i, data in enumerate(UD_tokenized):
        print(f"\t{['train', 'dev', 'test'][i]}: {np.mean([len(seq) for seq in data])} tokens")
    print(f"\ttrain+dev+test: {np.mean([len(seq) for data in UD_tokenized for seq in data])} tokens")


    for i, data in enumerate(CNEC_data):
        labels = [label for seq in data for label in seq["label"]]
        unique, counts = np.unique(labels, return_counts=True)
        print(f"CNEC {['train', 'dev', 'test'][i]}:")
        for u, c in zip(unique, counts):
            print(f"\t{u}: {c}")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.bar(unique, counts)
        ax.set_title(f"CNEC {['train', 'dev', 'test'][i]} label histogram")
        ax.set_xlabel("Label")
        plt.xticks()
        ax.set_ylabel("Count")
        plt.grid()
        plt.savefig(f"cv04/histogram_CNEC_{['train', 'dev', 'test'][i]}.svg")

    for i, data in enumerate(UD_data):
        labels = [label for seq in data for label in seq["label"]]
        unique, counts = np.unique(labels, return_counts=True)
        print(f"UD {['train', 'dev', 'test'][i]}:")
        for u, c in zip(unique, counts):
            print(f"\t{u}: {c}")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.bar(unique, counts)
        ax.set_title(f"UD {['train', 'dev', 'test'][i]} label histogram")
        ax.set_xlabel("Label")
        plt.xticks(rotation=30)
        ax.set_ylabel("Count")
        plt.grid()
        plt.savefig(f"cv04/histogram_UD_{['train', 'dev', 'test'][i]}.svg")


if __name__ == "__main__":
    main()
