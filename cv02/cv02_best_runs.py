import numpy as np
from scipy import stats


confidence = 0.90

def get_results(data, config_name):
    mean = np.mean(data)
    sem = stats.sem(data)  # standard error of the mean
    df = len(data) - 1

    interval = stats.t.interval(confidence, df, loc=mean, scale=sem)

    print(f"Configuration: {config_name}")
    print(f"Mean: {mean:.4f}")
    print(f"{int(confidence*100)}% confidence interval: ({interval[0]:.4f}, {interval[1]:.4f})")
    print()


# data obtained manually from wandb web interface

data = [
{"name": "c1", "scores": [1.7671, 1.92105, 1.84388, 1.95063, 2.01265, 1.65019, 1.71968, 1.96484, 1.63348, 1.5887]},
{"name": "c2", "scores": [1.94979, 1.87683, 1.73462, 1.75041, 1.79404, 1.84698, 1.91794, 1.92359, 1.78837, 1.69471]},
{"name": "c3", "scores": [1.9436, 2.04391, 2.18661, 1.80572, 1.99611, 1.90478, 1.79245, 2.02384, 1.97407, 1.61134]},
{"name": "c4", "scores": [2.30706, 1.85662, 1.85003, 1.8019, 2.3958, 2.18047, 1.88933, 1.96171, 2.19979, 1.6822]},
]

for d in data:
    get_results(d["scores"], d["name"])

#c1: batch_size=1000, emb_projection=true, emb_training=false, final_metric=neural, lr=0.1, lr_scheduler=exponential, optimizer=sgd, random_emb=true, vocab_size=30000
#c2: batch_size=500, emb_projection=true, emb_training=true, final_metric=neural, lr=0.1, lr_scheduler=exponential, optimizer=sgd, random_emb=true, vocab_size=20000
#c3: batch_size=1000, emb_projection=true, emb_training=true, final_metric=neural, lr=0.1, lr_scheduler=exponential, optimizer=sgd, random_emb=true, vocab_size=30000
#c4: batch_size=500, emb_projection=true, emb_training=false, final_metric=neural, lr=0.1, lr_scheduler=step, optimizer=sgd, random_emb=true, vocab_size=20000