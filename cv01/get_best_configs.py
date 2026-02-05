import wandb
import pandas as pd
import sys
import os

# Add the parent directory to the system path to import wandb_config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from wandb_config import WANDB_PROJECT, WANDB_ENTITY

api = wandb.Api()

project_path = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
runs = api.runs(project_path)


data = []
for run in runs:
    hyperparams = run.config
    test_acc = run.summary.get("test_acc")
    
    if test_acc is not None:
        row = hyperparams.copy()
        row["test_acc"] = test_acc
        data.append(row)

df = pd.DataFrame(data)

# Filter for 'dense' and 'cnn' models
for model_type in ['dense', 'cnn']:
    model_df = df[df['model'] == model_type]
    # Group by configuration except 'test_acc'
    group_cols = list(model_df.columns.difference(['test_acc']))
    grouped = model_df.groupby(group_cols)['test_acc']
    summary = grouped.agg(['mean', 'count', 'std']).reset_index()
    # Compute 90% confidence interval
    summary['ci90'] = 1.645 * summary['std'] / summary['count']**0.5
    # Sort by mean test_acc descending
    top2 = summary.sort_values('mean', ascending=False).head(2)
    print(f"\nTop 2 configurations for model == '{model_type}':")
    print(top2[[group_cols[1], group_cols[3], group_cols[4], group_cols[5]] + ['mean', 'ci90']])
