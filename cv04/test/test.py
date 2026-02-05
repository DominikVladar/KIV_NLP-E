import unittest
import torch
import json
import os

from my_utils import wandb_utils
from cv04.models import Czert, Slavic, RNN, LSTM
from cv04.main04 import lr_schedule


RESULTS = "results"


class TestCV04(unittest.TestCase):
    WandbData = None
    Results = {
        "CKPT1_Discussion": 0,
        "CKPT2": 0,
        "CKPT2_Discussion": "",
        "CKPT3": 0,
        "CKPT3_Discussion": "",
        "CKPT4": 0,
        "CKPT4_Discussion": "",
        "CKPT5": 0,
        "CKPT5_Discussion": "",
        "CKPT6": 0,
        "CKPT6_Discussion": "",
        "CKPT7": 0,
        "CKPT7_Discussion": "",
        "Sum_Tests": 0,
        "Overall": "",
    }

    MANDATORY_HP = ["task", "model_type"]
    MANDATORY_M = ["training_f1", "training_recall", "training_precision",
                   "eval_f1", "eval_recall", "eval_precision",
                   "test_f1", "test_recall", "test_precision"]

    NUM_LABELS = 10
    VOCAB_SIZE = 30_000

    @classmethod
    def setUpClass(cls):
        TestCV04.WandbData = wandb_utils.load_runs(["cv04"],
                                                   mandatory_hp=TestCV04.MANDATORY_HP,
                                                   mandatory_m=TestCV04.MANDATORY_M,
                                                   minimum_runtime_s=5,
                                                   unfold=True)

    @classmethod
    def tearDownClass(cls):
        TestCV04.Results["Sum_Tests"] = TestCV04.Results["CKPT2"] + TestCV04.Results["CKPT3"] + \
                                        TestCV04.Results["CKPT4"] + TestCV04.Results["CKPT5"] + \
                                        TestCV04.Results["CKPT6"] + TestCV04.Results["CKPT7"]

        formatted_results = dict()
        for i, (name, value) in enumerate(TestCV04.Results.items()):
            formatted_results[i] = (name, value)

        os.makedirs(RESULTS, exist_ok=True)
        with open(f"{RESULTS}/cv04.json", 'w', encoding="utf-8") as fd:
            json.dump(formatted_results, fd)

    def test_ckpt0_czert(self):
        model = Czert("UWB-AIR/Czert-B-base-cased",
                      torch.device("cpu"),
                      False,
                      0.01,
                      False,
                      0,
                      self.NUM_LABELS)
        out = model(torch.randint(0, self.VOCAB_SIZE, (8, 128)),
                    torch.randint(0, 1, (8, 128)),
                    torch.randint(0, 1, (8, 128)),
                    torch.randint(0, self.NUM_LABELS, (8, 128)))
        self.assertEqual(out.logits.shape[0], 8, "Model output has wrong shape")
        self.assertEqual(out.logits.shape[1], 128, "Model output has wrong shape")
        self.assertEqual(out.logits.shape[2], 10, "Model output has wrong shape")
        self.assertGreater(out.loss.item(), 0, "Loss was not computed correctly")

    def test_ckpt0_czert_freezing(self):
        to_freeze = 3
        model = Czert("UWB-AIR/Czert-B-base-cased",
                      torch.device("cpu"),
                      False,
                      0.01,
                      True,
                      3,
                      self.NUM_LABELS)
        freeze_layers = [f"layer.{i}." for i in range(to_freeze)]
        for name, param in model.named_parameters():
            if "embedding" in name:
                self.assertFalse(param.requires_grad, "Embedding layer is not frozen")
            if any(layer_pattern in name for layer_pattern in freeze_layers):
                self.assertFalse(param.requires_grad, "Transformer layer is not frozen")

    def test_ckpt0_slavic(self):
        model = Slavic("DeepPavlov/bert-base-bg-cs-pl-ru-cased",
                       torch.device("cpu"),
                       False,
                       0.01,
                       False,
                       0,
                       self.NUM_LABELS)
        out = model(torch.randint(0, self.VOCAB_SIZE, (8, 128)),
                    torch.randint(0, 1, (8, 128)),
                    torch.randint(0, 1, (8, 128)),
                    torch.randint(0, self.NUM_LABELS, (8, 128)))
        self.assertEqual(out.logits.shape[0], 8, "Model output has wrong shape")
        self.assertEqual(out.logits.shape[1], 128, "Model output has wrong shape")
        self.assertEqual(out.logits.shape[2], 10, "Model output has wrong shape")
        self.assertGreater(out.loss.item(), 0, "Loss was not computed correctly")

    def test_ckpt0_slavic_freezing(self):
        to_freeze = 2
        model = Slavic("DeepPavlov/bert-base-bg-cs-pl-ru-cased",
                       torch.device("cpu"),
                       False,
                       0.01,
                       True,
                       3,
                       self.NUM_LABELS)
        freeze_layers = [f"layer.{i}." for i in range(to_freeze)]
        for name, param in model.named_parameters():
            if "embedding" in name:
                self.assertFalse(param.requires_grad, "Embedding layer is not frozen")
            if any(layer_pattern in name for layer_pattern in freeze_layers):
                self.assertFalse(param.requires_grad, "Transformer layer is not frozen")

    def test_ckpt2_rnn(self):
        alpha = 0.02
        model = RNN(self.VOCAB_SIZE,
                    torch.device("cpu"),
                    128,
                    0.1,
                    False,
                    self.NUM_LABELS,
                    64,
                    False,
                    alpha)
        out = model(torch.randint(0, self.VOCAB_SIZE, (8, 128)),
                    torch.randint(0, 1, (8, 128)),
                    torch.randint(0, 1, (8, 128)),
                    torch.randint(0, self.NUM_LABELS, (8, 128)))
        self.assertEqual(out.logits.shape[0], 8, "Model output has wrong shape")
        self.assertEqual(out.logits.shape[1], 128, "Model output has wrong shape")
        self.assertEqual(out.logits.shape[2], 10, "Model output has wrong shape")
        self.assertGreater(out.loss.item(), 0, "Loss was not computed correctly")

        model = RNN(self.VOCAB_SIZE,
                    torch.device("cpu"),
                    128,
                    0.1,
                    False,
                    3,
                    64,
                    False,
                    alpha)
        out = model(torch.randint(0, self.VOCAB_SIZE, (4, 64)),
                    torch.randint(0, 1, (4, 64)),
                    torch.randint(0, 1, (4, 64)),
                    torch.randint(0, 3, (4, 64)))
        self.assertEqual(out.logits.shape[0], 4, "Model output has wrong shape")
        self.assertEqual(out.logits.shape[1], 64, "Model output has wrong shape")
        self.assertEqual(out.logits.shape[2], 3, "Model output has wrong shape")
        self.assertGreater(out.loss.item(), 0, "Loss was not computed correctly")
        self.Results["CKPT2"] += 2

    def test_ckpt4_rnn_freezing(self):
        self.assertTrue(self.Results["CKPT2"] > 0, "Cannot proceed to CKPT4 without implementing CKPT2")
        model = RNN(self.VOCAB_SIZE,
                    torch.device("cpu"),
                    128,
                    0.1,
                    True,
                    self.NUM_LABELS,
                    64,
                    False,
                    0.02)
        for name, param in model.named_parameters():
            if "embedding" in name:
                self.assertFalse(param.requires_grad, "Embedding layer is not frozen")
        self.Results["CKPT4"] += 0.5

    def test_ckpt4_rnn_l2(self):
        alpha = 0.02
        model = RNN(self.VOCAB_SIZE,
                    torch.device("cpu"),
                    128,
                    0.1,
                    True,
                    self.NUM_LABELS,
                    64,
                    False,
                    0.02)
        l2_norm = 0
        for param in model.named_parameters():
            if any(pattern in param[0] for pattern in ["hidden_state", "output_layer"]):
                l2_norm += torch.norm(param[1])
        l2_norm *= alpha
        self.assertEqual(model.compute_l2_norm(), l2_norm, "L2 norm was not computed correctly")
        self.Results["CKPT4"] += 0.5

    def test_ckpt2_rnn_structure(self):
        alpha = 0.02
        model = RNN(self.VOCAB_SIZE,
                    torch.device("cpu"),
                    128,
                    0.1,
                    False,
                    self.NUM_LABELS,
                    64,
                    False,
                    alpha)
        expected_modules = ['_embedding_layer', '_loss', '_dropout_layer',
                            '_new_hidden_state_layer', '_output_layer']
        for name, module in model.named_modules():
            if len(name) == 0:
                continue

            if expected_modules.count(name) > 0:
                expected_modules.remove(name)
            else:
                self.fail(f"Model contains redundant torch module: {name}")
            if '_embedding_layer' in name:
                self.assertTrue(isinstance(module, torch.nn.Embedding), "Embedding layer does not have correct class")
                self.assertEqual(128, module.embedding_dim, "Wrong embedding dimensions in RNN model")
                self.assertEqual(30000, module.num_embeddings, "Wrong vocab size in RNN model embedding layer")
            elif '_loss' in name:
                self.assertTrue(isinstance(module, torch.nn.CrossEntropyLoss), "Loss does not have correct class")
            elif '_dropout_layer' in name:
                self.assertTrue(isinstance(module, torch.nn.Dropout), "Dropout does not have correct class")
                self.assertEqual(0.1, module.p, "Dropout layer has a wrong dropout probability")
            elif '_new_hidden_state_layer' in name:
                self.assertTrue(isinstance(module, torch.nn.Linear),
                                "New hidden state layer does not have correct class")
                self.assertEqual(192, module.in_features,
                                 "New hidden state layer does not have correct number of input features")
                self.assertEqual(64, module.out_features,
                                 "New hidden state layer does not have correct number of output features")
            elif '_output_layer' in name:
                self.assertTrue(isinstance(module, torch.nn.Linear), "Output layer does not have correct class")
                self.assertEqual(64, module.in_features, "Output layer does not have correct number of input features")
                self.assertEqual(10, module.out_features,
                                 "Output layer does not have correct number of output features")
        self.assertEqual(0, len(expected_modules), "The model does not contain all the required modules")
        self.Results["CKPT2"] += 1.5

    def test_ckpt3_lstm(self):
        alpha = 0.02
        model = LSTM(self.VOCAB_SIZE,
                     128,
                     0.1,
                     False,
                     0,
                     self.NUM_LABELS,
                     2,
                     64,
                     False,
                     alpha)
        out = model(torch.randint(0, self.VOCAB_SIZE, (8, 128)),
                    torch.randint(0, 1, (8, 128)),
                    torch.randint(0, 1, (8, 128)),
                    torch.randint(0, self.NUM_LABELS, (8, 128)))
        self.assertEqual(out.logits.shape[0], 8, "Model output has wrong shape")
        self.assertEqual(out.logits.shape[1], 128, "Model output has wrong shape")
        self.assertEqual(out.logits.shape[2], 10, "Model output has wrong shape")
        self.assertGreater(out.loss.item(), 0, "Loss was not computed correctly")

        model = LSTM(self.VOCAB_SIZE,
                     32,
                     0.1,
                     False,
                     0,
                     3,
                     2,
                     128,
                     False,
                     alpha)
        out = model(torch.randint(0, self.VOCAB_SIZE, (4, 64)),
                    torch.randint(0, 1, (4, 64)),
                    torch.randint(0, 1, (4, 64)),
                    torch.randint(0, 3, (4, 64)))
        self.assertEqual(out.logits.shape[0], 4, "Model output has wrong shape")
        self.assertEqual(out.logits.shape[1], 64, "Model output has wrong shape")
        self.assertEqual(out.logits.shape[2], 3, "Model output has wrong shape")
        self.assertGreater(out.loss.item(), 0, "Loss was not computed correctly")
        self.Results["CKPT3"] += 1.5

    def test_ckpt4_lstm_freezing(self):
        self.assertTrue(self.Results["CKPT3"] > 0, "Cannot proceed to CKPT4 without implementing CKPT3")
        to_freeze = 1
        alpha = 0.02
        model = LSTM(self.VOCAB_SIZE,
                     128,
                     0.1,
                     True,
                     to_freeze,
                     self.NUM_LABELS,
                     2,
                     64,
                     False,
                     alpha)
        freeze_layers = [f"_l{i}" for i in range(to_freeze)]
        for name, param in model.named_parameters():
            if "embedding" in name:
                self.assertFalse(param.requires_grad, "Embedding layer is not frozen")
            if any(layer_pattern in name for layer_pattern in freeze_layers):
                self.assertFalse(param.requires_grad, "LSTM layer is not frozen")
        self.Results["CKPT4"] += 0.5

    def test_ckpt4_lstm_l2(self):
        alpha = 0.02
        model = LSTM(self.VOCAB_SIZE,
                     128,
                     0.1,
                     False,
                     0,
                     self.NUM_LABELS,
                     2,
                     64,
                     False,
                     alpha)
        l2_norm = 0
        for param in model.named_parameters():
            if any(pattern in param[0] for pattern in ["dense", "classification"]):
                l2_norm += torch.norm(param[1])
        l2_norm *= alpha
        self.assertEqual(model.compute_l2_norm(), l2_norm, "L2 norm was not computed correctly")
        self.Results["CKPT4"] += 0.5

    def test_ckpt3_lstm_structure(self):
        alpha = 0.02
        alpha = 0.02
        model = LSTM(self.VOCAB_SIZE,
                     128,
                     0.1,
                     False,
                     0,
                     self.NUM_LABELS,
                     2,
                     64,
                     False,
                     alpha)

        expected_modules = ['_embedding_layer', '_loss', '_lstm', '_dropout_layer',
                            '_dense', '_classification_head']
        for name, module in model.named_modules():
            if len(name) == 0:
                continue

            if expected_modules.count(name) > 0:
                expected_modules.remove(name)
            else:
                self.fail(f"Model contains redundant torch module: {name}")
            if '_lstm' in name:
                self.assertTrue(isinstance(module, torch.nn.LSTM), "LSTM layer does not have correct class")
                self.assertEqual(128, module.input_size, "Wrong input size of the LSTM layer")
                self.assertEqual(64, module.hidden_size, "Wrong hidden size of the LSTM layer")
                self.assertEqual(2, module.num_layers, "Wrong number of the LSTM layers")
            elif '_embedding_layer' in name:
                self.assertTrue(isinstance(module, torch.nn.Embedding), "Embedding layer does not have correct class")
                self.assertEqual(128, module.embedding_dim, "Wrong embedding dimensions in RNN model")
                self.assertEqual(30000, module.num_embeddings, "Wrong vocab size in RNN model embedding layer")
            elif '_loss' in name:
                self.assertTrue(isinstance(module, torch.nn.CrossEntropyLoss), "Loss does not have correct class")
            elif '_dropout_layer' in name:
                self.assertTrue(isinstance(module, torch.nn.Dropout), "Dropout does not have correct class")
                self.assertEqual(0.1, module.p, "Dropout layer has a wrong dropout probability")
            elif '_dense' in name:
                self.assertTrue(isinstance(module, torch.nn.Linear), "First dense layer has an incorrect class")
                self.assertEqual(128, module.in_features, "First dense layer has wrong input dimension")
                self.assertEqual(128, module.out_features, "First dense layer has wrong output dimension")
            elif '_classification_head' in name:
                self.assertTrue(isinstance(module, torch.nn.Linear), "Output layer does not have correct class")
                self.assertEqual(128, module.in_features, "Output layer does not have correct number of input features")
                self.assertEqual(10, module.out_features,
                                 "Output layer does not have correct number of output features")
        self.assertEqual(0, len(expected_modules), "The model does not contain all the required modules")
        self.Results["CKPT3"] += 1

    def test_ckpt5_lr_schedule(self):
        self.assertEqual(0.0, lr_schedule(0, 4000, 100000), "Wrong LR returned by lr_schedule()")
        self.assertEqual(1.0, lr_schedule(4000, 4000, 100000), "Wrong LR returned by lr_schedule()")
        self.assertEqual(0.0, lr_schedule(100000, 4000, 100000), "Wrong LR returned by lr_schedule()")
        self.assertEqual(0.5, lr_schedule(2000, 4000, 100000), "Wrong LR returned by lr_schedule()")
        self.assertEqual(0.5, lr_schedule(52000, 4000, 100000), "Wrong LR returned by lr_schedule()")
        self.assertEqual(0.5197916666666667, lr_schedule(50100, 4000, 100000), "Wrong LR returned by lr_schedule()")
        self.assertEqual(0.375, lr_schedule(1500, 4000, 100000), "Wrong LR returned by lr_schedule()")
        self.Results["CKPT5"] += 1

    def test_ckpt6_ner_experiments(self):
        experiments = [
            {"model_type": "LSTM", "task": "NER", "config.no_bias": True},
            {"model_type": "LSTM", "task": "NER", "config.no_bias": False},
            {"model_type": "LSTM", "task": "NER", "config.learning_rate": 0.0001},
            {"model_type": "LSTM", "task": "NER", "config.learning_rate": 0.001},
            {"model_type": "LSTM", "task": "NER", "config.l2_alpha": 0.01},
            {"model_type": "LSTM", "task": "NER", "config.l2_alpha": 0},
            {"model_type": "RNN", "task": "NER", "config.no_bias": True},
            {"model_type": "RNN", "task": "NER", "config.no_bias": False},
            {"model_type": "RNN", "task": "NER", "config.learning_rate": 0.0001},
            {"model_type": "RNN", "task": "NER", "config.learning_rate": 0.001},
            {"model_type": "RNN", "task": "NER", "config.l2_alpha": 0.01},
            {"model_type": "RNN", "task": "NER", "config.l2_alpha": 0},
            {"model_type": "CZERT",  "task": "NER"},
            {"model_type": "SLAVIC", "task": "NER"},
        ]

        # Check all experiments performed
        for experiment in experiments:
            self.assertTrue(wandb_utils.has_experiment(self.WandbData, **experiment),
                            f"Cannot find required experiment in wandb project - run: {experiment}")

        # Check RNN results
        self.assertTrue(wandb_utils.has_result_better_than(self.WandbData, 0.27, "summary.test_f1",
                                                           model_type="RNN", task="NER"))
        self.assertTrue(wandb_utils.has_result_better_than(self.WandbData, 0.27, "summary.test_recall",
                                                           model_type="RNN", task="NER"))
        self.assertTrue(wandb_utils.has_result_better_than(self.WandbData, 0.20, "summary.test_precision",
                                                           model_type="RNN", task="NER"))
        self.assertTrue(wandb_utils.has_result_less_than(self.WandbData, 2.5, "summary.test_loss",
                                                         model_type="RNN", task="NER"))

        # Check LSTM results
        self.assertTrue(wandb_utils.has_result_better_than(self.WandbData, 0.32, "summary.test_f1",
                                                           model_type="LSTM", task="NER"))
        self.assertTrue(wandb_utils.has_result_better_than(self.WandbData, 0.40, "summary.test_recall",
                                                           model_type="LSTM", task="NER"))
        self.assertTrue(wandb_utils.has_result_better_than(self.WandbData, 0.23, "summary.test_precision",
                                                           model_type="LSTM", task="NER"))
        self.assertTrue(wandb_utils.has_result_less_than(self.WandbData, 2.5, "summary.test_loss",
                                                         model_type="LSTM", task="NER"))

        self.assertTrue(wandb_utils.has_result_better_than(self.WandbData, 0.8, "summary.test_f1",
                                                           model_type="CZERT", task="NER"))
        self.assertTrue(wandb_utils.has_result_better_than(self.WandbData, 0.8, "summary.test_recall",
                                                           model_type="CZERT", task="NER"))
        self.assertTrue(wandb_utils.has_result_better_than(self.WandbData, 0.75, "summary.test_precision",
                                                           model_type="CZERT", task="NER"))
        self.assertTrue(wandb_utils.has_result_less_than(self.WandbData, 0.7, "summary.test_loss",
                                                         model_type="CZERT", task="NER"))

        self.assertTrue(wandb_utils.has_result_better_than(self.WandbData, 0.8, "summary.test_f1",
                                                           model_type="SLAVIC", task="NER"))
        self.assertTrue(wandb_utils.has_result_better_than(self.WandbData, 0.8, "summary.test_recall",
                                                           model_type="SLAVIC", task="NER"))
        self.assertTrue(wandb_utils.has_result_better_than(self.WandbData, 0.75, "summary.test_precision",
                                                           model_type="SLAVIC", task="NER"))
        self.assertTrue(wandb_utils.has_result_less_than(self.WandbData, 0.7, "summary.test_loss",
                                                         model_type="SLAVIC", task="NER"))
        self.Results["CKPT6"] += 1.5

    def test_ckpt6_tagging_experiments(self):
        experiments = [
            {"model_type": "LSTM", "task": "TAGGING", "config.no_bias": True},
            {"model_type": "LSTM", "task": "TAGGING", "config.no_bias": False},
            {"model_type": "LSTM", "task": "TAGGING", "config.learning_rate": 0.0001},
            {"model_type": "LSTM", "task": "TAGGING", "config.learning_rate": 0.001},
            {"model_type": "LSTM", "task": "TAGGING", "config.l2_alpha": 0.01},
            {"model_type": "LSTM", "task": "TAGGING", "config.l2_alpha": 0},
            {"model_type": "RNN", "task": "TAGGING", "config.no_bias": True},
            {"model_type": "RNN", "task": "TAGGING", "config.no_bias": False},
            {"model_type": "RNN", "task": "TAGGING", "config.learning_rate": 0.0001},
            {"model_type": "RNN", "task": "TAGGING", "config.learning_rate": 0.001},
            {"model_type": "RNN", "task": "TAGGING", "config.l2_alpha": 0.01},
            {"model_type": "RNN", "task": "TAGGING", "config.l2_alpha": 0},
            {"model_type": "CZERT",  "task": "TAGGING"},
            {"model_type": "SLAVIC", "task": "TAGGING"},
        ]

        # Check all experiments performed
        for experiment in experiments:
            self.assertTrue(wandb_utils.has_experiment(self.WandbData, **experiment),
                            f"Cannot find required experiment in wandb project - run: {experiment}")

        # Check RNN results
        self.assertTrue(wandb_utils.has_result_better_than(self.WandbData, 0.70, "summary.test_f1",
                                                           model_type="RNN", task="TAGGING"))
        self.assertTrue(wandb_utils.has_result_better_than(self.WandbData, 0.70, "summary.test_recall",
                                                           model_type="RNN", task="TAGGING"))
        self.assertTrue(wandb_utils.has_result_better_than(self.WandbData, 0.68, "summary.test_precision",
                                                           model_type="RNN", task="TAGGING"))
        self.assertTrue(wandb_utils.has_result_less_than(self.WandbData, 3, "summary.test_loss",
                                                         model_type="RNN", task="TAGGING"))

        # Check LSTM results
        self.assertTrue(wandb_utils.has_result_better_than(self.WandbData, 0.80, "summary.test_f1",
                                                           model_type="LSTM", task="TAGGING"))
        self.assertTrue(wandb_utils.has_result_better_than(self.WandbData, 0.80, "summary.test_recall",
                                                           model_type="LSTM", task="TAGGING"))
        self.assertTrue(wandb_utils.has_result_better_than(self.WandbData, 0.80, "summary.test_precision",
                                                           model_type="LSTM", task="TAGGING"))
        self.assertTrue(wandb_utils.has_result_less_than(self.WandbData, 3, "summary.test_loss",
                                                         model_type="LSTM", task="TAGGING"))

        self.assertTrue(wandb_utils.has_result_better_than(self.WandbData, 0.93, "summary.test_f1",
                                                           model_type="CZERT", task="TAGGING"))
        self.assertTrue(wandb_utils.has_result_better_than(self.WandbData, 0.93, "summary.test_recall",
                                                           model_type="CZERT", task="TAGGING"))
        self.assertTrue(wandb_utils.has_result_better_than(self.WandbData, 0.93, "summary.test_precision",
                                                           model_type="CZERT", task="TAGGING"))
        self.assertTrue(wandb_utils.has_result_less_than(self.WandbData, 0.2, "summary.test_loss",
                                                         model_type="CZERT", task="TAGGING"))

        self.assertTrue(wandb_utils.has_result_better_than(self.WandbData, 0.93, "summary.test_f1",
                                                           model_type="SLAVIC", task="TAGGING"))
        self.assertTrue(wandb_utils.has_result_better_than(self.WandbData, 0.93, "summary.test_recall",
                                                           model_type="SLAVIC", task="TAGGING"))
        self.assertTrue(wandb_utils.has_result_better_than(self.WandbData, 0.93, "summary.test_precision",
                                                           model_type="SLAVIC", task="TAGGING"))
        self.assertTrue(wandb_utils.has_result_less_than(self.WandbData, 0.2, "summary.test_loss",
                                                         model_type="SLAVIC", task="TAGGING"))
        self.Results["CKPT6"] += 1.5

    def test_ckpt7_experiments(self):
        experiments = [
            {"model_type": "CZERT", "task": "NER", "config.freeze_embedding_layer": True,
             "config.freeze_first_x_layers": 0},
            {"model_type": "CZERT", "task": "NER", "config.freeze_embedding_layer": True,
             "config.freeze_first_x_layers": 2},
            {"model_type": "CZERT", "task": "NER", "config.freeze_embedding_layer": True,
             "config.freeze_first_x_layers": 4},
            {"model_type": "CZERT", "task": "NER", "config.freeze_embedding_layer": True,
             "config.freeze_first_x_layers": 6},
            {"model_type": "BERT", "task": "NER"},
            {"model_type": "BERT", "task": "TAGGING"},
        ]

        # Check all experiments performed
        for experiment in experiments:
            self.assertTrue(wandb_utils.has_experiment(self.WandbData, **experiment),
                            f"Cannot find required experiment in wandb project - run: {experiment}")

        experiments_bert_ner = wandb_utils.get_experiments(self.WandbData, model_type="BERT", task="NER")
        experiments_bert_tagging = wandb_utils.get_experiments(self.WandbData, model_type="BERT", task="TAGGING")

        self.assertTrue(len(experiments_bert_ner.index) >= 5, "Not enough experiments with BERT on NER task")
        self.assertTrue(len(experiments_bert_tagging.index) >= 5, "Not enough experiments with BERT on TAGGING task")

        self.Results["CKPT7"] += 3
