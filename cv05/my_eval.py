from src.evaluator import eval_file
from pathlib import Path
import os
import matplotlib.pyplot as plt

csfd = {}
sts = {}
ner = {}


for file in os.listdir("submissions"):
    if file.endswith("csfd.json"):
        if "fixed" not in file:
            continue
        model = file.split("submission_")[1].split("_new_query")[0].split(".csfd")[0]
        model = model[:model.rfind("_")]
        if model.startswith("meta_llama"):
            model = "Meta Llama 3.1 70B Instruct"
        elif model.startswith("openai"):
            model = "OpenAI GPT 4o Mini"
        else:
            model = "Qwen 2.5 72B Instruct"
        if model not in csfd:
            csfd[model] = eval_file(Path(f"submissions/{file}"))["accuracy"]
        else:
            score = eval_file(Path(f"submissions/{file}"))["accuracy"]
            csfd[model] = max(csfd[model], score)
    elif file.endswith("sts.json"):
        model = file.split("submission_")[1].split("_new_query")[0].split(".sts")[0]
        model = model[:model.rfind("_")]
        if model.startswith("meta_llama"):
            model = "Meta Llama 3.1 8B"
        elif model.startswith("openai"):
            model = "OpenAI GPT 4o Mini"
        elif model.startswith("x_ai"):
            model = "X AI Grok 4.1 Fast"
        else:
            if "72" in model:
                model = "Qwen 2.5 72B Instruct"
            else:
                model = "Qwen 2.5 7B Instruct"

        if model not in sts:
            sts[model] = eval_file(Path(f"submissions/{file}"))["mse"]
        else:
            score = eval_file(Path(f"submissions/{file}"))["mse"]
            sts[model] = min(sts[model], score)

    else:
        if "with_tags" not in file:
            continue
        model = file.split("submission_")[1].split("_with_tags")[0].split(".ner")[0]
        model = model[:model.rfind("_")]
        if model.startswith("meta_llama"):
            model = "Meta Llama 3.1 70B Instruct"
        elif model.startswith("openai"):
            model = "OpenAI GPT 4o Mini"
        elif model.startswith("x_ai"):
            model = "X AI Grok 4.1 Fast"
        else:
            model = "Qwen 2.5 72B Instruct"
        if model not in ner:
            ner[model] = eval_file(Path(f"submissions/{file}"))["f1-macro"]
        else:
            score = eval_file(Path(f"submissions/{file}"))["f1-macro"]
            ner[model] = max(ner[model], score)


ner = dict(sorted(ner.items(), key=lambda x: x[1]))
sts = dict(sorted(sts.items(), key=lambda x: x[1]))


os.makedirs("output", exist_ok=True)
plt.figure(figsize=(8, 6))
plt.bar(csfd.keys(), csfd.values())
plt.xlabel("Model")
plt.ylabel("Accuracy")
for i, val in enumerate(csfd.values()):
    plt.text(i-0.07, val + 0.005, f"{val:.2f}", fontsize=12)
plt.savefig("output/csfd.png")

plt.figure(figsize=(11, 6))
plt.bar(sts.keys(), sts.values())
plt.xlabel("Model")
plt.ylabel("MSE")
for i, val in enumerate(sts.values()):
    plt.text(i-0.1, val + 0.01, f"{val:.2f}", fontsize=12)
plt.savefig("output/sts.png")


plt.figure(figsize=(11, 6))
plt.bar(ner.keys(), ner.values())
plt.xlabel("Model")
plt.ylabel("F1 macro")
for i, val in enumerate(ner.values()):
    plt.text(i-0.07, val + 0.005, f"{val:.2f}", fontsize=12)
plt.savefig("output/ner.png")