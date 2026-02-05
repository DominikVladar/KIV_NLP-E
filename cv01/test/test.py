from enum import Enum
import json
import os
import statistics
from unittest import TestCase

import my_utils.wandb_utils as wandb_utils

ACC_MINIMUM_N = 10
ACC_MINIMUM_VAL = 99

RESULTS = "results"

class Column(Enum):
    DATA_HIST = 0
    RAND_MAJ = 1
    EXPS = 2
    META = 3
    ParCoor = 4
    ShowRuns = 5
    Table = 6
    NExp = 7
    BestAcc = 8
    ConfDivRuns = 9
    Disc = 10

class TestMnist(TestCase):
    results = dict()

    @classmethod
    def setUpClass(cls):
        for c in Column:
            TestMnist.add_result_value(c, "-")

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

        with open(f"{RESULTS}/cv01.json", 'w', encoding="utf-8") as fd:
            json.dump(TestMnist.results, fd)

    def __init__(self, *args, **kwargs):
        super(TestMnist, self).__init__(*args, **kwargs)
        mandatory_hp = ["lr", "optimizer", "dp"]
        mandatory_m = ["test_acc", "train_loss", "test_loss"]
        self.wandb_data = wandb_utils.load_runs(["cv01"], mandatory_hp=mandatory_hp, mandatory_m=mandatory_m)

    def test_grid(self):
        grid = {"lr": [0.1, 0.01, 0.001, 0.0001, 0.00001], "model": ["dense", "cnn"], "optimizer": ["sgd", "adam"], "dp": [0, 0.1, 0.3, 0.5]}
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
            TestMnist.add_result_value(Column.EXPS, "NEE")

            self.fail("FAILED")
        TestMnist.add_result_value(Column.EXPS, 3)

    def test_acc(self):
        
        test_acc = wandb_utils.best_metric(self.wandb_data, "test_acc")
        acc_top_10_mean = statistics.mean(test_acc[:10])
        if len(test_acc) < ACC_MINIMUM_N:
            TestMnist.add_result_value(Column.BestAcc, 0)
            TestMnist.add_result_value(Column.NExp, "NEE")
            

            self.fail(f"too little experiments {len(test_acc)} < {ACC_MINIMUM_N}")
        if test_acc[ACC_MINIMUM_N - 1] < ACC_MINIMUM_VAL:
            TestMnist.add_result_value(Column.BestAcc, acc_top_10_mean)
            TestMnist.add_result_value(Column.NExp, "LOW")

            self.fail(
                f"not found satisfactory results\ntop {ACC_MINIMUM_N} test_acc:{test_acc[:ACC_MINIMUM_N]} should be > {ACC_MINIMUM_VAL}")
        
        TestMnist.add_result_value(Column.BestAcc, acc_top_10_mean)
        TestMnist.add_result_value(Column.NExp, "OK")        
        


        

