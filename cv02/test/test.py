import json
import os
import statistics
import unittest
import time
import torch

import cv02.main02
from cv02.consts import TRAIN_DATA, EMB_FILE, TEST_DATA

from cv02.main02 import dataset_vocab_analysis, load_ebs, MySentenceVectorizer, DataLoader, DummyModel, test,WORD2IDX,VECS_BUFF
from my_utils import wandb_utils

FINAL_MINIMUM_N = 10
FINAL_MAXIMUM_VAL = 2

TEST_BATCH_SIZE = 3
TEST_MAX_SEQ_LEN = 13

RESULTS = "results"
class WandbTest(unittest.TestCase):
    EXP_LEN = 273378
    EXP_TOP = []
    results = None

    @classmethod
    def setUpClass(cls):
        WandbTest.results = dict()

    @classmethod
    def tearDownClass(cls):
        os.makedirs(RESULTS, exist_ok=True)

        with open(f"{RESULTS}/cv02.json", 'w', encoding="utf-8") as fd:
            json.dump(WandbTest.results, fd)

    def __init__(self, *args, **kwargs):
        super(WandbTest, self).__init__(*args, **kwargs)
        with open(TRAIN_DATA, 'r', encoding="utf-8") as fd:
            self.train_data_texts = fd.read().split("\n")

        mandatory_hp = ["random_emb", "emb_training", "emb_projection", "vocab_size", "final_metric", "lr", "optimizer", "batch_size"]
        mandatory_m = ["train_loss", "test_loss"]

        self.wandb_data = wandb_utils.load_runs(["cv02"], mandatory_hp=mandatory_hp, mandatory_m=mandatory_m, minimum_runtime_s=10)

    def clear_cache(self):
        if os.path.exists(WORD2IDX):
            os.remove(WORD2IDX)
        if os.path.exists(VECS_BUFF):
            os.remove(VECS_BUFF)

    def test_ckpt_01(self):
        top_n_words = dataset_vocab_analysis(self.train_data_texts, -1)

        if abs(len(top_n_words) - WandbTest.EXP_LEN) > 0:
            WandbTest.results[0] = ("ckpt_1","F")
            self.fail(f"you have different size of the vocabulary")
        else:
            print("vocab size is ok")
            WandbTest.results[0] = ("ckpt_1","ok")

    def test_ckpt_02(self):
        ## todo measure time
        self.clear_cache()

        load_ebs(EMB_FILE, dataset_vocab_analysis(self.train_data_texts, -1), 20000)
        exists = os.path.exists(cv02.main02.WORD2IDX) and os.path.exists(cv02.main02.VECS_BUFF)
        if not exists:
            WandbTest.results[1] = ("ckpt_2", "f")
            self.fail(f"files does not exists")
        else:
            print("files exist")
            WandbTest.results[1] = ("ckpt_2", "ok")

    def test_ckpt_03(self):

        self.clear_cache()

        start = time.time()


        top_n_words = dataset_vocab_analysis(self.train_data_texts, -1)

        word2idx, word_vectors = load_ebs(EMB_FILE, top_n_words, 20000)
        vectorizer = MySentenceVectorizer(word2idx,TEST_MAX_SEQ_LEN)

        inp = "Podle vlády dnes není dalších otázek"
        EXPECTED = [259, 642, 249, 66, 252, 3226]
        vectorized = vectorizer.sent2idx(inp)[:len(EXPECTED)]
        print(vectorized)
        if EXPECTED != vectorized:
            WandbTest.results[2] = ("ckpt_3", "f")
            self.fail("bad ids")
        else:
            WandbTest.results[2] = ("ckpt_3", "ok")
            print("ids are ok")

        end = time.time()

        print(f"Time for ststs+embs+vec : {end-start:.2f}s")
        WandbTest.results[7] = ("stats_time", f"{end - start:.2f}s")



    def test_ckpt_04(self):
        top_n_words = dataset_vocab_analysis(self.train_data_texts, -1)
        word2idx, word_vectors = load_ebs(EMB_FILE, top_n_words, 20000)
        vectorizer = MySentenceVectorizer(word2idx, TEST_MAX_SEQ_LEN)


        train_dataset = DataLoader(vectorizer, TRAIN_DATA, TEST_BATCH_SIZE)
        test_dataset = DataLoader(vectorizer, TEST_DATA, TEST_BATCH_SIZE)

        print(f"loaded train:{len(train_dataset.sts)}\tout of vocab:{train_dataset.out_of_vocab} %")
        print(f"loaded test:{len(test_dataset.sts)}\tout of vocab:{test_dataset.out_of_vocab} %")

        # todo check random shuffeling
        one_sample = next(iter(train_dataset))
        # print(f"returning {type(one_sample)}\nlen:{len(one_sample)} expected(3)\n")
        exp_a = [TEST_BATCH_SIZE, TEST_MAX_SEQ_LEN]
        exp_sts = [TEST_BATCH_SIZE]
        if list(one_sample["a"].shape) != exp_a:
            WandbTest.results[3] = ("ckpt_4", "f")
            self.fail(f"tensor a has wrong shape {list(one_sample['a'].shape)}  ... expected {exp_a}")
        if list(one_sample["b"].shape) != exp_a:
            WandbTest.results[3] = ("ckpt_4", "f")
            self.fail(f"tensor b has wrong shape {list(one_sample['b'].shape)}  ... expected {exp_a}")
        if list(one_sample["sts"].shape) != exp_sts:
            WandbTest.results[3] = ("ckpt_4", "f")
            self.fail(f"tensor sts has wrong shape {list(one_sample['sts'].shape)}  ... expected {exp_sts}")

        WandbTest.results[3] = ("ckpt_4", "ok")


    def test_ckpt_05(self):
        top_n_words = dataset_vocab_analysis(self.train_data_texts, -1)
        word2idx, word_vectors = load_ebs(EMB_FILE, top_n_words, 20000)
        vectorizer = MySentenceVectorizer(word2idx, TEST_MAX_SEQ_LEN)

        train_dataset = DataLoader(vectorizer, TRAIN_DATA, TEST_BATCH_SIZE)
        test_dataset = DataLoader(vectorizer, TEST_DATA, TEST_BATCH_SIZE)

        device = "cpu"
        dummy_net = DummyModel(train_dataset)
        dummy_net = dummy_net.to(device)

        loss_function = torch.nn.MSELoss()

        dummy_test = test(test_dataset, dummy_net, loss_function)
        dummy_train = test(train_dataset, dummy_net, loss_function)

        # assert abs(dummy_test-3.1970975200335183) < 0.1
        if abs(dummy_test - 3.197097628880292) > 1:
            WandbTest.results[4] = ("dummy", "f")
            self.fail(f"Your model probably does not return correct value .. wrong test value")
        if abs(dummy_train - 4.496822993359776) > 1:
            WandbTest.results[4] = ("dummy", "f")
            self.fail(f"Your model probably does not return correct value .. wrong train value")

        WandbTest.results[4] = ("dummy", "5")

    def test_grid(self):
        grid = {"lr": [0.01, 0.001, 0.0001, 0.00001],
                "optimizer": ["sgd", "adam"],
                "random_emb": [True, False],
                "emb_training": [True, False],
                "emb_projection": [True, False],
                "final_metric": ["cos", "neural"],
                "vocab_size": [20_000],

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

    def test_final(self):
        test_acc = wandb_utils.best_metric(self.wandb_data, "test_loss", sort_invert=False)
        acc_top_10_mean = statistics.mean(test_acc[:10])


        if len(test_acc) < FINAL_MINIMUM_N:
            WandbTest.results[6] = ("MSE_pt", "NEE")
            self.fail(f"too little experiments {len(test_acc)} < {FINAL_MINIMUM_N}")

        if test_acc[FINAL_MINIMUM_N - 1] > FINAL_MAXIMUM_VAL:
            WandbTest.results[6] = ("MSE_pt", "LOW")
            self.fail(
                f"not found satisfactory results\ntop {FINAL_MINIMUM_N} test_loss:{test_acc[:FINAL_MINIMUM_N]} should be < {FINAL_MAXIMUM_VAL}")

        WandbTest.results[6] = ("MSE_pt", "10")
        WandbTest.results[6] = ("MSE_VAL", f"{acc_top_10_mean}")


# if __name__ == "__main__":
#     unittest.main()
    # a = MyTest()
    # a.test_final()
