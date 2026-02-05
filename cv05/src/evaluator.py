"""
Evaluator for the fifth NLP-E homework exercise.

Format specification:
    Loads and evaluates files with double suffix `.<taskname>.json(l)`.
    Both JSON and JSONLINES formats are accepted, but JSON files must
    have a list as the root object.

    Task name is one of: "csfd", "sts", "ner"

    The format of items in submission files is:
    {"testset_id": <sample index in test set>, "prediction": <task prediction>}

    Both additional fields and additional items are allowed, it is recommended
    to store raw predictions and experiment metadata with prompt templates if possible.

    Correct data types for the prediction are: csfd - int, sts - float, ner - list[str]

Usage:
    When run as a script, this evaluator reads the "submissions" directory and reports best results.

    The evaluator always assumes it is being ran from the assignment directory for dataset access.

    Or you can pass a list a specific file to evaluate as the first program argument.

    Among all found submissions for each task, the best results are displayed.
"""
from pathlib import Path
from seqeval.metrics import f1_score
import json
import sys

def _load_predictions(path: Path) -> list[dict]:
    """ Loads .json or .jsonl file with predictions. """
    if path.suffix.lower().endswith("json"):
        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON data *must* be a list!")
    elif path.suffix.lower().endswith("jsonl"):
        data = []
        invalid_lines = 0

        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except ValueError:
                    invalid_lines += 1
            
            if invalid_lines:
                total_lines = len(data) + invalid_lines
                print(f"Skipped {invalid_lines} out of {total_lines} in {path!s}")
    else:
        raise ValueError(f"File is not in supported format (.json or .jsonl): {path}")
    return data

def _order_predictions(
        predictions: list[dict],
        max_id: int,
        pred_field="prediction",
        id_field="testset_id"
    ) -> list:
    """ Orders predictions in testset order. """
    preds = {p[id_field]: p for p in predictions if id_field in p}
    consecutive = [preds.get(idx, {}).get(pred_field) for idx in range(max_id)]
    return consecutive

def load_tsv(path: Path, header=None, map_fields={}) -> list[dict]:
    """ Load tsv file, with optional header specification and field mapping. """
    data = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            toks = line.strip().split("\t")

            if not header:
                header = toks
                continue
            
            # this might deserve nicer error handling, but its solvable by passing non-raising functions
            row = {k: map_fields.get(k, lambda x: x)(v) for k, v in zip(header, toks)}
            
            data.append(row)
    
    return data

def load_ner(path: Path) -> list[dict[str, list]]:
    """ Load net text format, return list of sentences. """
    data = []
    toks, tags = [], []

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            
            if not line:
                data.append({"tokens": toks, "tags": tags})
                toks, tags = [], []
                continue

            vals = line.split()
            if len(vals) > 2:
                vals = [vals[0], vals[-1]]

            tok, tag = vals

            toks.append(tok)
            tags.append(tag)
        
    data.append({"tokens": toks, "tags": tags})

    return data

def eval_csfd(*, targets: list[int], predictions: list[int]) -> dict[str, float]:
    """ Evaluate ČSFD. Targets and predictions are both lists of integers 0-2. """
    correct = sum(p == t for p, t, in zip(predictions, targets))
    total = max(len(predictions), len(targets))

    return {
        "accuracy": correct / total
    }

def eval_sts(*, targets: list[float], predictions: list[float]) -> dict[str, float]:
    """ Evaluate STS. Targets and predictions are both lists of floats 0.0-6.0 """
    # make sure no Nones on input
    predictions = [p if p is not None else 2.5 for p in predictions]

    square_error = sum((p - t)**2 for p, t, in zip(predictions, targets))
    total = len(targets)

    return {
        "mse": square_error / total
    }

def eval_ner(*, targets: list[list[str]], predictions: list[list[str]]) -> dict[str, float]:
    """ Evaluate NER. Targets and predictions are both lists of lists of tags. """
    for idx in range(len(predictions)):
        if predictions[idx]:
            continue
        # allow for missing values
        predictions[idx] = ["O"] * len(targets[idx])
    
    return {
        "f1-micro": f1_score(targets, predictions, average="micro"),
        "f1-macro": f1_score(targets, predictions, average="macro"),
    } # type: ignore (f1_score is typed as -> list[float] | float, but will return float)

def eval_file(submission: Path):
    if len(submission.suffixes) < 2:
        raise ValueError(f"File {submission} is not a valid submission, expected suffixes `.<taskname>.json(l)`.")

    taskname = submission.suffixes[-2][1:]

    predictions = _load_predictions(submission)

    try:
        if taskname == "csfd":
            data = load_tsv(Path("data/csfd-test-llm.tsv"), map_fields={"label": int})
            predictions = _order_predictions(predictions, len(data))
            targets = [x["label"] for x in data]
            return eval_csfd(targets=targets, predictions=predictions)

        elif taskname == "sts":
            data = load_tsv(
                Path("data/anlp01-sts-free-test-llm.tsv"),
                header=["a", "b", "sts"],
                map_fields={"sts": float}
            )
            predictions = _order_predictions(predictions, len(data))
            targets = [x["sts"] for x in data]
            return eval_sts(targets=targets, predictions=predictions)

        elif taskname == "ner":
            data = load_ner(Path("data/ner-dev-llm.txt"))
            predictions = _order_predictions(predictions, len(data))
            targets = [x["tags"] for x in data]
            return eval_ner(targets=targets, predictions=predictions)

        else:
            raise ValueError(f"Unknown task name: {taskname}")
    except Exception as ex:
        print(f"Failed to evaluate submission \"{submission!s}\" due to:")
        print(f"{ex.__class__.__name__}: {ex!s}\n")
        return {}


def main():
    if len(sys.argv) > 2:
        path = Path(sys.argv[1])

        if not path.exists() or not path.is_file():
            print(f"File to evaluate {sys.argv[1]} must a be a file!")
            return {}
        
        metrics = eval_file(path)
        print(metrics)
        return metrics

    all_submissions = Path("submissions").glob("**/*.json")

    best_results = {}
    best_submissions = {}

    for path in all_submissions:
        if len(path.suffixes) < 2 or path.suffixes[-2] not in [".csfd", ".sts", ".ner"]:
            continue

        taskname = path.suffixes[-2][1:]
        metrics = eval_file(path)
        best_metrics = best_results.get(taskname, {})

        for metric in metrics:
            value = metrics[metric]
            best = best_metrics.get(metric, None)
            if best is not None:
                if metric in ["mse"]: # lower is better metrics
                    best_metrics[metric] = min(best, value)
                else: # default is higher is better
                    best_metrics[metric] = max(best, value)
                if best != best_metrics[metric]:
                    best_submissions[f"{taskname}-{metric}"] = path
            else:
                best_metrics[metric] = value
                best_submissions[f"{taskname}-{metric}"] = path
        
        best_results[taskname] = best_metrics

    if not best_results:
        print("No submissions found.")
        return

    print("Best results:")
    
    for task, results in best_results.items():
        for metric, value in results.items():
            print(f"{task}-{metric}: {value:.5f}")
    
    print("\nBest submissions:")
    for taskmetric, path in best_submissions.items():
            print(f"{taskmetric}: {path}")

if __name__ == "__main__":
    main()
