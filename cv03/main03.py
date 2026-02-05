# https://pytorch.org/data/main/tutorial.html
# https://towardsdatascience.com/text-classification-with-cnns-in-pytorch-1113df31e79f
import configparser
import os
import pickle
import sys
from collections import defaultdict
import json

import random
import torch
# from ignite.contrib.handlers.param_scheduler import create_lr_scheduler_with_warmup
from datasets import load_dataset

import wandb
from torch import nn
from torch.utils.data import DataLoader

import wandb_config
from cv02.consts import EMB_FILE
from cv02.main02 import dataset_vocab_analysis, MySentenceVectorizer, PAD, UNK

import numpy as np
from math import floor
from collections import Counter

NUM_CLS = 3

CNN_MODEL = "cnn"
MEAN_MODEL = "mean"

CSFD_DATASET_TRAIN = "cv03/data/csfd-train.tsv"
CSFD_DATASET_TEST = "cv03/data/csfd-test.tsv"

WORD2IDX = "word2idx.pckl"
VECS_BUFF = "vecs.pckl"

CLS_NAMES = ["neg", "neu", "pos"]

from wandb_config import WANDB_PROJECT, WANDB_ENTITY


def count_statistics(dataset, vectorizer):
    ## todo CF#01

    vectorizer.reset_counter()
    for t in dataset["text"]:
        vectorizer.sent2idx(t)

    coverage = floor((1 - (vectorizer.out_of_vocab_perc() / 100))*10)/10
    class_distribution = Counter(dataset["label"])
    class_distribution = {k: v / len(dataset["label"]) for k, v in class_distribution.items()}

    return coverage, class_distribution


class MyBaseModel(torch.nn.Module):

    def __init__(self, config, w2v=None):
        super(MyBaseModel, self).__init__()
        self.config = config

        self.softmax = nn.Softmax(dim=-1)
        self.activation = nn.GELU() if config["activation"] == "gelu" else nn.ReLU()
        if w2v is not None:
            self.emb_layer = nn.Embedding.from_pretrained(torch.tensor(w2v, dtype=torch.float32), padding_idx=0, freeze=not config["emb_training"])
            self.emb_proj = nn.Linear(w2v.shape[1], config["proj_size"])
        else:
            config["emb_size"] = 300
            self.emb_layer = nn.Embedding(config["vocab_size"], config["emb_size"], padding_idx=0)
            self.emb_proj = nn.Linear(config["emb_size"], config["proj_size"])

class MyModelAveraging(MyBaseModel):
    def __init__(self, config, w2v=None):
        super(MyModelAveraging, self).__init__(config,w2v)
        # head input dim must match the actual embedding projection used in forward
        if getattr(self, "emb_proj", None) is not None:
            head_in_dim = config["proj_size"]
        else:
            # if no projection layer, use original embedding size
            head_in_dim = np.array(w2v).shape[1] if w2v is not None else config.get("emb_size", 300)
        self.head = nn.Linear(head_in_dim, NUM_CLS)

    def forward(self, x, l=None):
        x = torch.tensor(x).to(self.config["device"])
        emb = self.emb_layer(x).float()
        if self.emb_proj is not None:
            emb = self.emb_proj(emb)
            a = self.activation(emb)
        res = torch.mean(emb, dim=1)

        return self.softmax(self.head(res))


class MyModelConv(MyBaseModel):
    def __init__(self, config, w2v=None):
        super(MyModelConv, self).__init__(config, w2v),
        self.config = config

        # todo CF#CNN_CONF
        self.cnn_architecture = config["cnn_architecture"]  # for unit tests
        if self.cnn_architecture == "A":
            self.cnn_config = [(1, config["n_kernel"], (2, 1)),
                               (1, config["n_kernel"], (3, 1)),
                               (1, config["n_kernel"], (4, 1))]
            self.config["hidden_size"] = 500
        elif self.cnn_architecture == "B":
            self.config["hidden_size"] = 970
            self.cnn_config = [(1, config["n_kernel"], (2, config["proj_size"] // 2)),
                               (1, config["n_kernel"], (3, config["proj_size"] // 2)),
                               (1, config["n_kernel"], (4, config["proj_size"] // 2))]
        elif self.cnn_architecture == "C":
            self.config["hidden_size"] = 35020
            self.cnn_config = [(1, config["n_kernel"], (2, config["proj_size"])),
                               (1, config["n_kernel"], (3, config["proj_size"])),
                               (1, config["n_kernel"], (4, config["proj_size"]))]
        self.conv_layers = []
        for i, (in_channels, out_channels, kernel_size) in enumerate(self.cnn_config):
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size))
        self.max_pools = [nn.MaxPool2d((config["seq_len"] - kernel_size[2][0] + 1, 1)) for kernel_size in
                          self.cnn_config]
        reduced_emb_size = config["proj_size"] if config["emb_projection"] else w2v.shape[1]
        reduced_emb_size = reduced_emb_size - self.cnn_config[0][2][1] + 1
        self.proj = nn.Linear(reduced_emb_size * len(self.cnn_config) * config["n_kernel"], self.config["hidden_size"])
        self.head = nn.Linear(self.config["hidden_size"], NUM_CLS)

        self.dropout = nn.Dropout(0.5)

        # !!!!!! TODO
        # this line is important if you use list to group your architecture ... optimizer would not register if it is not used
        self.modules = nn.ModuleList(self.conv_layers)

    def forward(self, x, l=None):
        x = torch.tensor(x).to(self.config["device"])
        x = self.emb_layer(x)
        x = x.float()
        if self.config["emb_projection"]:
            x = self.emb_proj(x)
            x = self.activation(x)
        x = x.unsqueeze(1)
        conv_outs = []
        for i, conv in enumerate(self.conv_layers):
            conv_out = conv(x)
            conv_out = self.activation(conv_out)
            conv_out = self.max_pools[i](conv_out)
            conv_out = self.dropout(conv_out)
            conv_outs.append(conv_out)
        x = torch.cat(conv_outs, dim=1)
        x = x.view(x.shape[0], -1)
        x = self.proj(x)
        x = self.activation(x)
        x = self.head(x)
        return self.softmax(x)


def test_on_dataset(dataset_iterator, vectorizer, model, loss_metric_func, config):
    test_loss_list = []
    test_acc_list = []
    test_enum_y = []
    test_enum_pred = []
    with torch.no_grad():
        for b in dataset_iterator:
            texts = b["text"]
            labels = torch.tensor(b["label"], dtype=torch.long).to(config["device"])

            vectorized = [vectorizer.sent2idx(x) for x in texts]

            pred = model(vectorized)
            loss = loss_metric_func(pred, labels)
            test_loss_list.append(loss.item())

            pred_cls = torch.argmax(pred, dim=-1)
            acc = torch.sum(pred_cls == labels) / len(labels)
            test_acc_list.append(acc.item())

            test_enum_y.append(labels)
            test_enum_pred.append(pred_cls)

    return {
        "test_acc": sum(test_acc_list) / len(test_acc_list),
        "test_loss": sum(test_loss_list) / len(test_loss_list),
        "test_pred_clss": torch.cat(test_enum_pred),
        "test_enum_gold": torch.cat(test_enum_y),
    }


def main(config: dict):
    cls_dataset = load_dataset("csv", delimiter='\t', data_files={"train": [CSFD_DATASET_TRAIN],
                                                                  "test": [CSFD_DATASET_TEST]})

    top_n_words = dataset_vocab_analysis(cls_dataset['train']["text"], top_n=-1)
    print("Top N words:", len(top_n_words))

    word2idx, word_vectors = load_ebs(EMB_FILE, top_n_words, config['vocab_size'], force_rebuild=False)

    vectorizer = MySentenceVectorizer(word2idx, config["seq_len"])

    coverage, cls_dist = count_statistics(cls_dataset['train'], vectorizer)
    print(f"COVERAGE: {coverage}\ncls_dist:{cls_dist}")

    slit_dataset = cls_dataset['train'].train_test_split(test_size=0.2, shuffle=True)
    cls_dataset['train'] = slit_dataset['train']
    cls_dataset['valid'] = slit_dataset['test']

    cls_train_iterator = DataLoader(cls_dataset['train'], batch_size=config['batch_size'])
    cls_val_iterator = DataLoader(cls_dataset['valid'], batch_size=config['batch_size'])
    cls_test_iterator = DataLoader(cls_dataset['test'], batch_size=config['batch_size'])

    if config["model"] == CNN_MODEL:
        model = MyModelConv(config, w2v=word_vectors)
    elif config["model"] == MEAN_MODEL:
        model = MyModelAveraging(config, w2v=word_vectors)

    num_of_params = 0
    for x in model.parameters():
        print(x.shape)
        num_of_params += torch.prod(torch.tensor(x.shape), 0)
    config["num_of_params"] = int(num_of_params)
    print("num of params:", num_of_params)

    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, tags=["cv03"], config=config)

    model.to(config["device"])
    cross_entropy = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

    batch = 0
    while True:
        for b in cls_train_iterator:
            texts = b["text"]
            labels = torch.tensor(b["label"], dtype=torch.long).to(config["device"])

            vectorized = [vectorizer.sent2idx(x) for x in texts]

            optimizer.zero_grad()
            pred = model(vectorized)

            loss = cross_entropy(pred, labels)
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                model.eval()
                ret = test_on_dataset(cls_val_iterator, vectorizer, model, cross_entropy, config)
                conf_matrix = wandb.plot.confusion_matrix(preds=ret["test_pred_clss"].cpu().numpy(),
                                                          y_true=ret["test_enum_gold"].cpu().numpy(),
                                                          class_names=CLS_NAMES)
                wandb.log({"conf_mat": conf_matrix})
                #
                wandb.log({"test_acc": ret["test_acc"],
                           "test_loss": ret["test_loss"]}, commit=False)
                model.train()
                print(f"batch: {batch} test_acc: {ret['test_acc']} test_loss: {ret['test_loss']}")

            train_acc = torch.sum(torch.argmax(pred, dim=-1) == labels) / len(labels)
            total_norm = nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            wandb.log(
                {"train_loss": loss, "train_acc": train_acc, "lr": lr_scheduler.get_last_lr()[0], "pred": pred,
                 "norm": total_norm})
            batch += 1

        lr_scheduler.step()

        if batch >= config["batches"]:
            break

    ret = test_on_dataset(cls_test_iterator, vectorizer, model, cross_entropy, config)
    wandb.log({"final_test_acc": ret["test_acc"],
               "final_test_loss": ret["test_loss"]})


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print(f"Using {device}")

    N_KERNEL = 64
    PROJ_SIZE = 100

    config = {"batch_size": 33,
              "vocab_size": 20000,
              # "model": MEAN_MODEL,
              "model": CNN_MODEL,
              "device": device,
              "n_kernel": N_KERNEL,

              "activation": "relu",
              "random_emb": False,
              "emb_training": False,
              "emb_projection": True,
              "proj_size": PROJ_SIZE,
              "cnn_architecture": "C",

              "emb_size": 300,
              "lr": 0.0005,
              "gradient_clip": .5,
              'batches': 200000,
              "seq_len": 100
              }

    main(config)



def load_ebs(emb_file, top_n_words: list, wanted_vocab_size, force_rebuild=True, random_emb=False):
    print("prepairing W2V...", end="")
    if random_emb:
        print("...random embeddings")
        word2idx = {}
        for i, w in enumerate(top_n_words[:wanted_vocab_size]):
            word2idx[w] = i
        word2idx[UNK] = len(word2idx)
        word2idx[PAD] = len(word2idx)
        vecs = np.random.uniform(-1, 1, (wanted_vocab_size + 2, 300))
        vecs[word2idx[PAD]] = np.zeros(300)
        assert len(vecs) == len(word2idx)

        return word2idx, vecs
    if os.path.exists(WORD2IDX) and os.path.exists(VECS_BUFF) and not force_rebuild:
        # CF#3
        print("...loading from buffer")
        with open(WORD2IDX, 'rb') as idx_fd, open(VECS_BUFF, 'rb') as vecs_fd:
            word2idx = pickle.load(idx_fd)
            vecs = pickle.load(vecs_fd)
    else:
        print("...creating from scratch")

        with open(emb_file, 'r', encoding="utf-8") as emb_fd:
            word2idx = {}
            vecs = []

            for idx, word in enumerate(top_n_words[:wanted_vocab_size]):
                word2idx[word] = idx

            vecs = np.random.uniform(-1, 1, (len(word2idx) + 2, 300))
            for i, l in enumerate(emb_fd):
                if i == 0:
                    continue
                l = l.strip().split(" ")
                word = l[0]
                if word in word2idx:
                    vecs[word2idx[word]] = np.array(l[1:], dtype=np.float32)

            word2idx[UNK] = len(word2idx)
            word2idx[PAD] = len(word2idx)
            vecs[word2idx[PAD]] = np.zeros(300)

            pickle.dump(word2idx, open(WORD2IDX, 'wb'))
            pickle.dump(vecs, open(VECS_BUFF, 'wb'))

    return word2idx, vecs
