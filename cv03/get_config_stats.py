from scipy import stats
import numpy as np


with open("cv03/wandb_runs.csv", "r") as f:
    lines = f.readlines()
    header = lines[0]
    lines = lines[1:] 
    cnn_architecture_index = header.split(",").index("\"cnn_architecture\"")
    model_index = header.split(",").index("\"model\"")
    test_acc_index = header.split(",").index("\"final_test_acc\"")
    test_accs = {"mean": [], "cnn_a": [], "cnn_b": [], "cnn_c": []}
    test_accs_best = {"mean": [], "cnn_a": [], "cnn_b": [], "cnn_c": []}
    for line in lines:
        b = 1 if "best" in line else 0
        try:
            acc = float(line.split(",")[test_acc_index+b][1:-1])
        except:
            continue
        model = line.split(",")[model_index+b]
        cnn_architecture = line.split(",")[cnn_architecture_index+b]
        if model == "\"mean\"":
            test_accs["mean"].append(acc)
            if b == 1:
                test_accs_best["mean"].append(acc)
        else:
            test_accs["cnn_" + cnn_architecture[1].lower()].append(acc)
            if b == 1:
                test_accs_best["cnn_" + cnn_architecture[1].lower()].append(acc)

    for model in ["mean", "cnn_a", "cnn_b", "cnn_c"]:
        for data in [test_accs[model], test_accs_best[model]]:
            mean = np.mean(data)
            # vypocet 90% konf. intervalu pomoci t-rozdeleni
            confidence = 0.90
            n = len(data)
            std_err = stats.sem(data)  # standard error
            h = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)


            print(f"{model}: {mean:.3f} +- {h:.3f}")