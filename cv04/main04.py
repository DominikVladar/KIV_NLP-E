import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from wandb_config import WANDB_PROJECT, WANDB_ENTITY

import numpy as np
import wandb
import torch
import datetime
from seqeval.metrics import f1_score, precision_score, recall_score
from sklearn import metrics as sklearn_metrics
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from transformers import (
    BertTokenizerFast,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from cv04.ner_utils import NerDataset, Split, get_labels
from cv04.models import Czert, Slavic, LSTM, RNN

import warnings

warnings.simplefilter("ignore")

logger = logging.getLogger(__name__)
MODEL_TYPES = ["RNN", "LSTM", "CZERT", "SLAVIC", "BERT"]

labels = None
label_map = None
compute_metrics = None

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_type: str = field(
        metadata={"help": "Model type from MODEL_TYPES"}
    )
    task: str = field(
        default="NER", metadata={"help": "Task name in ['NER', 'TAGGING']"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded"}
    )
    random_init: Optional[bool] = field(default=False, metadata={"help": "Use randomly initialized weights."})

    dropout_probs: Optional[float] = field(default=0.2, metadata={"help": "Dropout probability"})

    freeze_embedding_layer: Optional[bool] = field(default=False,
                                                   metadata={"help": "Freeze embedding layer of the model."})

    freeze_first_x_layers: Optional[int] = field(default=0,
                                                 metadata={"help": "Freeze x bottom layers of the model."})

    embedding_dimension: Optional[int] = field(default=128, metadata={"help": "Embedding dimension for RNN and LSTM."})
    num_lstm_layers: Optional[int] = field(default=2, metadata={"help": "Number of BiLSTM layers in the LSTM model."})
    lstm_hidden_dimension: Optional[int] = field(default=128, metadata={"help": "LSTM hidden dimension LSTM model."})
    no_bias: Optional[bool] = field(default=False, metadata={"help": "Use bias in cls head of RNN and LSTM model."})
    l2_alpha: Optional[float] = field(default=0.05, metadata={"help": "L2 regularization factor."})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .txt files for a CoNLL-2003-formatted task."}
    )
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a file containing all labels. If not specified, CoNLL-2003 labels are used."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    ),
    eval_dataset_batches: Optional[int] = field(
        default=None, metadata={"help": "How many batches from eval dataset to use"}
    )


def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
    batch_size, seq_len = predictions.shape

    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                out_label_list[i].append(label_map[label_ids[i][j]])
                preds_list[i].append(label_map[predictions[i][j]])

    return preds_list, out_label_list


def compute_metrics_ner(model_predictions, true_labels) -> Dict:
    model_predictions, true_labels = align_predictions(model_predictions, true_labels)
    return {
        "precision": precision_score(true_labels, model_predictions),
        "recall": recall_score(true_labels, model_predictions),
        "f1": f1_score(true_labels, model_predictions),
    }


def compute_metrics_tagging(model_predictions, true_labels) -> Dict:
    model_predictions, true_labels = align_predictions(model_predictions, true_labels)
    model_predictions = [item for sublist in model_predictions for item in sublist]
    true_labels = [item for sublist in true_labels for item in sublist]
    return {
        "precision": sklearn_metrics.precision_score(true_labels, model_predictions, average="macro"),
        "recall": sklearn_metrics.recall_score(true_labels, model_predictions, average="macro"),
        "f1": sklearn_metrics.f1_score(true_labels, model_predictions, average="macro"),
    }


def lr_schedule(step: int, warmup_steps: int, total_steps: int) -> float:
    """
    Learning rate schedule function that is used in torch.optim.lr_scheduler.LambdaLR

    The schedule first computes a linear warmup from 0% of the learning rate to 100% of the learning rate
    during first INIT_LR_STEPS steps. Then it computes a linear decay from 100% to 0% for the rest of the training.
    :param total_steps: Total number of learning steps to be performed = (dataset_size * num_train_epochs)
    :param warmup_steps: Number of warmup steps for linear start of the LR
    :param step: current training step
    :return: factor for LR calculation using formula 'lr * factor'
    """
    # TODO START
    # Compute a factor for LR at the current step in interval [0, 1]
    # Use linear warmup during warmup_steps and then a linear decay to 0 (from warmup_steps to total_steps)
    # return a float number in [0, 1]
    return step/warmup_steps if step < warmup_steps else (total_steps-step)/(total_steps-warmup_steps)
    # TODO END


def main():
    global labels
    global label_map
    global compute_metrics

    # See all possible arguments at
    # https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
    # WARNING: not all of them are utilized in the assignment
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    compute_metrics = lambda pred, lab: compute_metrics_ner(pred, lab) if model_args.task == "NER" else compute_metrics_tagging(pred, lab)

    # Check the selected model type
    if model_args.model_type not in MODEL_TYPES:
        logger.critical(f"Model type {model_args.model_type} is not available. "
                        f"Please select one of [{', '.join(MODEL_TYPES)}]!")
        exit(-1)

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed - numpy, torch, tf
    set_seed(training_args.seed)

    # Prepare label information
    labels = get_labels(data_args.labels)
    label_map = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)

    # Load pretrained model and tokenizer
    model = None
    tokenizer = None
    if model_args.model_type == "RNN":
        tokenizer = BertTokenizerFast.from_pretrained("UWB-AIR/Czert-B-base-cased")
        model = RNN(tokenizer.vocab_size,
                    device,
                    model_args.embedding_dimension,
                    model_args.dropout_probs,
                    model_args.freeze_embedding_layer,
                    num_labels,
                    model_args.lstm_hidden_dimension,
                    not model_args.no_bias,
                    model_args.l2_alpha
                    )
    if model_args.model_type == "LSTM":
        tokenizer = BertTokenizerFast.from_pretrained("UWB-AIR/Czert-B-base-cased")
        model = LSTM(tokenizer.vocab_size,
                     model_args.embedding_dimension,
                     model_args.dropout_probs,
                     model_args.freeze_embedding_layer,
                     model_args.freeze_first_x_layers,
                     num_labels,
                     model_args.num_lstm_layers,
                     model_args.lstm_hidden_dimension,
                     not model_args.no_bias,
                     model_args.l2_alpha
                     )
    if model_args.model_type == "SLAVIC":
        tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/bert-base-bg-cs-pl-ru-cased")
        model = Slavic("DeepPavlov/bert-base-bg-cs-pl-ru-cased",
                       device,
                       model_args.random_init,
                       model_args.dropout_probs,
                       model_args.freeze_embedding_layer,
                       model_args.freeze_first_x_layers,
                       num_labels)
    if model_args.model_type == "CZERT":
        tokenizer = BertTokenizerFast.from_pretrained("UWB-AIR/Czert-B-base-cased")
        model = Czert("UWB-AIR/Czert-B-base-cased",
                      device,
                      model_args.random_init,
                      model_args.dropout_probs,
                      model_args.freeze_embedding_layer,
                      model_args.freeze_first_x_layers,
                      num_labels)
    # TODO BONUS START
    # if model_args.model_type == "BERT" instantiate a CZERT model and BertTokenizerFast
    # in the same way as above just using "bert-base-cased" as a first argument for both
    # (instead of "UWB-AIR/Czert-B-base-cased")
    if model_args.model_type == "BERT":
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
        model = Czert("bert-base-cased", device, model_args.random_init, model_args.dropout_probs, model_args.freeze_embedding_layer, model_args.freeze_first_x_layers, num_labels)
    # TODO BONUS END

    model.to(device)
    model.train()

    config = model.get_config()
    for arg_class in [training_args, data_args, model_args]:
        attributes = [attr for attr in filter(lambda a: not a.startswith('__'), dir(arg_class))]
        for attribute in attributes:
            wandb.config[attribute] = getattr(arg_class, attribute)

    # Get datasets
    train_dataset = (
        NerDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type if hasattr(config, "model_type") else config["model_type"],
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        NerDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type if hasattr(config, "model_type") else config["model_type"],
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.dev,
        )
        if training_args.do_eval
        else None
    )

    # TRAIN LOOP
    do_train(model, tokenizer, train_dataset, eval_dataset, training_args, data_args)

    if training_args.do_predict:
        test_dataset = NerDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type if hasattr(config, "model_type") else config["model_type"],
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.test,
        )
        do_test(model, test_dataset, training_args, data_args)


def do_test(model, test_dataset, training_args, data_args):
    global labels
    global compute_metrics
    logger.info("Starting test")
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=training_args.per_device_eval_batch_size,
                                              shuffle=True,
                                              drop_last=True)

    test_labels = []
    test_predictions = []

    with torch.no_grad():
        running_precision_test = 0.0
        running_recall_test = 0.0
        running_loss_test = 0.0
        running_f1_test = 0.0
        test_batches = 0

        for test_ex in test_loader:
            test_ex = test_ex.to(device)
            test_batches += 1
            input_ids, attention_mask, token_type_ids, batch_labels = torch.unbind(test_ex, dim=1)
            model_out = model(input_ids=input_ids.contiguous(),
                              attention_mask=attention_mask.contiguous(),
                              token_type_ids=token_type_ids.contiguous(),
                              labels=batch_labels.contiguous())
            loss = model_out.loss
            predictions = torch.argmax(model_out.logits.cpu(), dim=-1)

            test_predictions.extend(predictions.cpu().numpy().tolist())
            test_labels.extend(batch_labels.cpu().numpy().tolist())

            batch_metrics = compute_metrics(np.array(predictions.cpu()), np.array(batch_labels.cpu()))
            running_precision_test += batch_metrics["precision"]
            running_recall_test += batch_metrics["recall"]
            running_loss_test += loss.item()
            running_f1_test += batch_metrics["f1"]

        logging_precision_test = running_precision_test / test_batches
        logging_recall_test = running_recall_test / test_batches
        logging_loss_test = running_loss_test / test_batches
        logging_f1_test = running_f1_test / test_batches

        logger.info(f"TEST - "
                    f"loss={logging_loss_test:.3f}, "
                    f"precision={logging_precision_test:.3f}, "
                    f"recall={logging_recall_test:.3f}, "
                    f"f1={logging_f1_test:.3f}")

        wandb.run.summary["test_loss"] = logging_loss_test
        wandb.run.summary["test_precision"] = logging_precision_test
        wandb.run.summary["test_recall"] = logging_recall_test
        wandb.run.summary["test_f1"] = logging_f1_test

    # Save predictions
    predictions_list, labels_list = align_predictions(np.array(test_predictions), np.array(test_labels))
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir, exist_ok=True)
    output_test_predictions_file = os.path.join(training_args.output_dir,
                                                f"test_predictions-{str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%s'))}.txt")
    with open(output_test_predictions_file, "w") as writer:
        with open(os.path.join(data_args.data_dir, "test.txt"), "r") as f:
            example_id = 0
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    writer.write(line)
                    if not predictions_list[example_id]:
                        example_id += 1
                elif predictions_list[example_id]:
                    output_line = line.split()[0] + " " + predictions_list[example_id].pop(0) + " " \
                                  + labels_list[example_id].pop(0) + "\n"
                    writer.write(output_line)


def do_train(model, tokenizer, train_dataset, eval_dataset, training_args, dataset_args):
    global labels
    global compute_metrics
    optimizer = torch.optim.Adam(model.parameters(), lr=training_args.learning_rate)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: lr_schedule(step * training_args.per_device_train_batch_size,
                                                                       training_args.warmup_steps,
                                                                       int(training_args.num_train_epochs) * len(
                                                                           train_dataset)))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=training_args.per_device_train_batch_size,
                                               shuffle=True,
                                               drop_last=True)

    for epoch in range(int(training_args.num_train_epochs)):
        logger.info(f"Starting epoch {epoch + 1}")
        running_precision = 0.0
        running_recall = 0.0
        running_loss = 0.0
        running_f1 = 0.0
        for batch, ex in enumerate(train_loader, 1):
            ex = ex.to(device)
            input_ids, attention_mask, token_type_ids, batch_labels = torch.unbind(ex, dim=1)

            optimizer.zero_grad()
            model_out = model(input_ids=input_ids.contiguous(),
                              attention_mask=attention_mask.contiguous(),
                              token_type_ids=token_type_ids.contiguous(),
                              labels=batch_labels.contiguous())
            l2_norm = model.compute_l2_norm()
            loss = model_out.loss + l2_norm
            predictions = torch.argmax(model_out.logits.cpu(), dim=-1)
            loss.backward()
            optimizer.step()
            scheduler.step()

            batch_metrics = compute_metrics(np.array(predictions.cpu()), np.array(batch_labels.cpu()))
            running_precision += batch_metrics["precision"]
            running_recall += batch_metrics["recall"]
            running_loss += loss.item()
            running_f1 += batch_metrics["f1"]

            if batch % training_args.logging_steps == 0:
                logging_precision = running_precision / training_args.logging_steps
                logging_recall = running_recall / training_args.logging_steps
                logging_loss = running_loss / training_args.logging_steps
                logging_f1 = running_f1 / training_args.logging_steps
                running_precision = 0
                running_recall = 0
                running_loss = 0
                running_f1 = 0

                logger.info(f"Batch {batch}: loss={logging_loss:.3f}, precision={logging_precision:.3f}, "
                            f"recall={logging_recall:.3f}, f1={logging_f1:.3f}")

                wandb.log(
                    {"training_loss": logging_loss,
                     "training_precision": logging_precision,
                     "training_recall": logging_recall,
                     "training_f1": logging_f1,
                     "lr": optimizer.state_dict()["param_groups"][0]["lr"],
                     "epoch": epoch + 1},
                    step=(epoch + 1) * len(train_dataset) + batch * training_args.per_device_train_batch_size)

            if batch % training_args.eval_steps == 0:
                model.eval()
                do_eval(batch, dataset_args, epoch, eval_dataset, model, tokenizer, train_dataset, training_args)
                model.train()
    logger.info("Finished training")


def do_eval(batch, dataset_args, epoch, eval_dataset, model, tokenizer, train_dataset, training_args):
    global labels
    global compute_metrics
    logger.info("Starting evaluation")
    eval_loader = torch.utils.data.DataLoader(eval_dataset,
                                              batch_size=training_args.per_device_eval_batch_size,
                                              shuffle=True,
                                              drop_last=True)
    with torch.no_grad():

        running_precision_eval = 0.0
        running_recall_eval = 0.0
        running_loss_eval = 0.0
        running_f1_eval = 0.0
        eval_batches = 0

        for eval_ex in eval_loader:
            eval_batches += 1
            eval_ex = eval_ex.to(device)
            input_ids, attention_mask, token_type_ids, batch_labels = torch.unbind(eval_ex, dim=1)
            model_out = model(input_ids=input_ids.contiguous(),
                              attention_mask=attention_mask.contiguous(),
                              token_type_ids=token_type_ids.contiguous(),
                              labels=batch_labels.contiguous())
            loss = model_out.loss
            predictions = torch.argmax(model_out.logits.cpu(), dim=-1)

            batch_metrics = compute_metrics(np.array(predictions.cpu()), np.array(batch_labels.cpu()))
            running_precision_eval += batch_metrics["precision"]
            running_recall_eval += batch_metrics["recall"]
            running_loss_eval += loss.item()
            running_f1_eval += batch_metrics["f1"]

            if dataset_args.eval_dataset_batches is not None and eval_batches >= dataset_args.eval_dataset_batches:
                logger.info(f"Stopping eval after {eval_batches} batches")
                break
        logging_precision_eval = running_precision_eval / eval_batches
        logging_recall_eval = running_recall_eval / eval_batches
        logging_loss_eval = running_loss_eval / eval_batches
        logging_f1_eval = running_f1_eval / eval_batches

        logger.info(f"EVAL - Batch {batch}: "
                    f"loss={logging_loss_eval:.3f}, "
                    f"precision={logging_precision_eval:.3f}, "
                    f"recall={logging_recall_eval:.3f}, "
                    f"f1={logging_f1_eval:.3f}")

        wandb.log(
            {"eval_loss": logging_loss_eval,
             "eval_precision": logging_precision_eval,
             "eval_recall": logging_recall_eval,
             "eval_f1": logging_f1_eval,
             "epoch": epoch + 1},
            step=(epoch + 1) * len(train_dataset) + batch * training_args.per_device_train_batch_size)

        print_sample_output(input_ids, batch_labels, predictions, tokenizer)


def print_sample_output(input_ids, labels, predictions, tokenizer):
    aligned_predictions, aligned_labels = align_predictions(predictions.cpu().numpy(), labels.cpu().numpy())
    print_label = aligned_labels[0]
    print_label = [l + " " * (3 - len(l)) for l in print_label]
    print_predictions = aligned_predictions[0]
    print_predictions = [p + " " * (3 - len(p)) for p in print_predictions]
    print_input_ids = input_ids.cpu()[0].numpy().tolist()
    decoded_input = tokenizer.decode(print_input_ids)
    logger.info(f"Sample input:  {decoded_input}")
    logger.info(f"Sample labels: {' '.join(print_label)}")
    logger.info(f"Sample preds:  {' '.join(print_predictions)}")


if __name__ == "__main__":
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, tags=["cv04"])

    wandb.define_metric("training_loss", summary="min")
    wandb.define_metric("training_precision", summary="max")
    wandb.define_metric("training_recall", summary="max")
    wandb.define_metric("training_f1", summary="max")

    wandb.define_metric("eval_loss", summary="min")
    wandb.define_metric("eval_precision", summary="max")
    wandb.define_metric("eval_recall", summary="max")
    wandb.define_metric("eval_f1", summary="max")

    main()

