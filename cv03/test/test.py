import json
import os
import statistics
import unittest
import time
import logging

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import sem

import cv02.main02
from cv02.consts import TRAIN_DATA, EMB_FILE, TEST_DATA

from cv02.main02 import dataset_vocab_analysis, MySentenceVectorizer, DataLoader, DummyModel, test, WORD2IDX, \
    VECS_BUFF
from cv03.main03 import CSFD_DATASET_TRAIN, CSFD_DATASET_TEST, count_statistics, MyModelConv, load_ebs
from my_utils import wandb_utils
from enum import Enum

# Configure logging to DEBUG level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

FINAL_MINIMUM_N = 10
# FINAL_MINIMUM_N = 2

TEST_BATCH_SIZE = 3
TEST_MAX_SEQ_LEN = 13



# todo pridat stdev do final metric

class Column(Enum):
    W_FREQ = 0
    COVERAGE = 1
    CLS_DIST = 2
    MEAN_PT = 3
    MEAN_ACC = 4
    CNN_A_ACC = 5
    CNN_B_ACC = 6
    CNN_C_ACC = 7


RESULTS = "results"


def count_parameters(model):
    params_total = 0
    for parameter in model.parameters():
        if not parameter.requires_grad: continue
        print(parameter.shape)
        params = parameter.numel()
        params_total += params
    return params_total


MANDATORY_HP = ["activation", "model", "random_emb", "emb_training", "emb_projection", "lr", "proj_size", "batch_size"]
MANDATORY_HP_CNN = ["cnn_architecture", "n_kernel", "hidden_size"]

MANDATORY_HP = ["activation", "model", "random_emb", "emb_training", "emb_projection", "proj_size", "batch_size"]

MANDATORY_M = ["train_acc", "test_loss", "train_loss"]


class WandbTest(unittest.TestCase):
    EXP_TOP = []
    results = None

    @classmethod
    def setUpClass(cls):
        WandbTest.results = dict()

    @classmethod
    def add_result_value(cls, col: Column, val):
        if isinstance(val, float):
            if val < 1:
                val = f"{val:.4f}"
            else:
                val = f"{val:.2f}"
        cls.results[col.value] = (col.name, val)

    @classmethod
    def tearDownClass(cls):
        os.makedirs(RESULTS, exist_ok=True)

        with open(f"{RESULTS}/cv03.json", 'w', encoding="utf-8") as fd:
            json.dump(WandbTest.results, fd)

    def __init__(self, *args, **kwargs):
        super(WandbTest, self).__init__(*args, **kwargs)
        with open(TRAIN_DATA, 'r', encoding="utf-8") as fd:
            self.train_data_texts = fd.read().split("\n")

        self.wandb_data = wandb_utils.load_runs(["cv03"], mandatory_hp=MANDATORY_HP, mandatory_m=MANDATORY_M, minimum_runtime_s=0)

    def clear_cache(self):
        if os.path.exists(WORD2IDX):
            os.remove(WORD2IDX)
        if os.path.exists(VECS_BUFF):
            os.remove(VECS_BUFF)

    def test_word_freq_analysis(self):
        EXP_LEN = 273378
        top_n_words = dataset_vocab_analysis(self.train_data_texts, -1)

        if abs(len(top_n_words) - EXP_LEN) > 5:
            WandbTest.add_result_value(Column.W_FREQ, "f")
            self.fail(f"you have different results of word analysis in train dataset")
        else:
            WandbTest.add_result_value(Column.W_FREQ, "ok")

    def test_clss_dist(self):
        VOCAB_SIZE = 20000
        SEQ_LEN = 100
        cls_dataset = load_dataset("csv", delimiter='\t', data_files={"train": [CSFD_DATASET_TRAIN],
                                                                      "test": [CSFD_DATASET_TEST]})

        all_train_texts = []
        for s in iter(cls_dataset['train']):
            all_train_texts.append(s["text"])

        top_n_words = dataset_vocab_analysis(all_train_texts, VOCAB_SIZE)
        word2idx, word_vectors = load_ebs(EMB_FILE, top_n_words, VOCAB_SIZE)
        vectorizer = MySentenceVectorizer(word2idx, SEQ_LEN)

        coverage, cls_dist = count_statistics(cls_dataset["train"], vectorizer)

        if abs(coverage - 0.73) > 0.05:
            WandbTest.add_result_value(Column.COVERAGE, "f")
            self.fail(f"you have different coverage")
        else:
            WandbTest.add_result_value(Column.COVERAGE, coverage)
            print("coverage is ok")

        expected = {0: 0.3251, 2: 0.3380, 1: 0.3367}
        for c in range(3):
            if abs(cls_dist[c] - expected[c]) > 0.0001:
                WandbTest.add_result_value(Column.CLS_DIST, "f")
                self.fail(f"wrong cls distribution")
        WandbTest.add_result_value(Column.CLS_DIST, "OK")

    def test_embedding_matrix_size(self):
        EXP_VOCAB_SIZE = 20000
        ## todo measure time
        # self.clear_cache()

        word2idx, vecs = load_ebs(EMB_FILE, dataset_vocab_analysis(self.train_data_texts, -1), EXP_VOCAB_SIZE)
        if abs(len(word2idx) - EXP_VOCAB_SIZE) > 5:
            WandbTest.results[1] = ("ckpt_2", "f")
            self.fail(f"Wrong vocab size : {len(word2idx)} expected cca {EXP_VOCAB_SIZE}")

        exists = os.path.exists(cv02.main02.WORD2IDX) and os.path.exists(cv02.main02.VECS_BUFF)
        if not exists:
            WandbTest.results[1] = ("ckpt_2", "f")
            self.fail(f"files does not exists")
        else:
            print("files exist")
            WandbTest.results[1] = ("ckpt_2", "ok")

    def test_cnn_grid(self):
        grid = {"lr": [0.001, 0.0001, 0.00001, 0.000001],
                "model": ["cnn"],
                "activation": ["gelu", "relu"],
                "random_emb": [True, False],
                "emb_training": [True, False],
                "emb_projection": [True],
                "cnn_config": ["A", "B", "C"],
                }
        min_n = 2

        grid_status = wandb_utils.grid_status(self.wandb_data, grid)

        ok = True
        for config, num_runs in grid_status.items():
            if num_runs < min_n:
                print(f"not enough experiments with configuration {config} (only {num_runs}<{min_n})")
                ok = False
            else:
                print(f"configuration {config} (runs:{num_runs}) -> OK")
        if not ok:
            WandbTest.results[5] = ("grid", "f")
            self.fail("FAILED")
        WandbTest.results[5] = ("grid", "ok")

    def test_parameters_number(self):

        config = {"n_kernel": 300,
                  "vocab_size": 20000,
                  "hidden_size": 0,
                  "activation": "relu",
                  "random_emb": True,
                  "emb_training": True,
                  "emb_projection": True,
                  "proj_size": 70,
                  "cnn_architecture": "A",
                  "device": "cpu",
                  "emb_size": 100,
                  "seq_len": 100,
                  "dropout": 0
                  }

        params = []
        config["n_kernel"] = 300

        modela = MyModelConv(config)
        params.append(count_parameters(modela))

        config["cnn_architecture"] = "B"

        # config["hidden_size"] = int(((63000/61200)) * hidden)
        modelb = MyModelConv(config)
        params.append(count_parameters(modelb))

        config["cnn_architecture"] = "C"
        config["n_kernel"] = 300
        # config["hidden_size"] = int((63000/(config["n_kernel"]*3))*hidden)
        modelc = MyModelConv(config)
        params.append(count_parameters(modelc))

        params = [p - (config["vocab_size"] * config["emb_size"] + config["emb_size"] * config["emb_projection"]) for p
                  in params]
        print(params)

        stdev = np.std(params)
        stdevs = [stdev / p for p in params]
        print(stdevs)
        if max(stdevs) > 0.005:
            self.fail("models have big difference between parameters number")

    def check_best(self, filtered_data, min_acc, top_runs: int, metric="test_acc", col_metric: Column = None,
                   col_pt: Column = None, val_pt=None):
        test_acc = wandb_utils.best_metric(filtered_data, metric, top_n=top_runs)
        print(test_acc)
        if not test_acc:
            if col_pt is not None:
                WandbTest.add_result_value(col_pt, "NEE")
            self.fail(f"too little experiments {len(test_acc)} < {top_runs} found: {test_acc}")

        if test_acc[-1] < min_acc:
            if col_pt is not None:
                WandbTest.add_result_value(col_pt, "LOW")
            self.fail(
                f"not found satisfactory results\n all top {top_runs} test_loss:{test_acc} should be > {min_acc}")

        if col_metric is not None:
            acc_top_n_mean = statistics.mean(test_acc)
            conf_interval = sem(test_acc)*2.2622 #for 10 examples and 95conf

            WandbTest.add_result_value(col_metric, f"{acc_top_n_mean}(±{conf_interval})")
        if col_pt is not None:
            WandbTest.add_result_value(col_pt, val_pt)

    def test_final_model_mean(self):

        data = wandb_utils.load_runs(["cv03"], mandatory_hp=MANDATORY_HP, mandatory_m=MANDATORY_M,
                                     minimum_runtime_s=0, best_only=True)

        filtered = wandb_utils.filter_data(data, {"model": "mean"})

        print("original:", self.wandb_data)
        print("filter:", filtered)

        self.check_best(filtered, 0.7, FINAL_MINIMUM_N, col_metric=Column.MEAN_ACC, col_pt=Column.MEAN_PT, val_pt=5)

    def final_cnn_analysis(self, architecture, col: Column, min_acc):

        data = wandb_utils.load_runs(["cv03"], mandatory_hp=MANDATORY_HP + MANDATORY_HP_CNN,
                                     mandatory_m=MANDATORY_M,
                                     minimum_runtime_s=0, best_only=True)
        filtered = wandb_utils.filter_data(data, {"model": "cnn", "cnn_architecture": architecture})

        print("original:", data)
        print("filter:", filtered)

        self.check_best(filtered, min_acc, FINAL_MINIMUM_N, col_metric=col)

    def test_final_cnn_A(self):
        self.final_cnn_analysis("A", Column.CNN_A_ACC, 0.6)

    def test_final_cnn_B(self):
        self.final_cnn_analysis("B", Column.CNN_B_ACC, 0.6)

    def test_final_cnn_C(self):
        self.final_cnn_analysis("C", Column.CNN_C_ACC, 0.6)

