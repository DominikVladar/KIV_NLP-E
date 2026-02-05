import sys
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import wandb
import itertools

from wandb_config import WANDB_PROJECT, WANDB_ENTITY

import logging

# Cache configuration
CACHE_DIR = Path.home() / ".cache" / "wandb_runs"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_cache_hash(tags):
    """Generate a hash for the cache key based on tags only."""
    params = {
        "tags": sorted(tags) if tags else None,
    }
    params_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()


def _get_cache_hash_with_params(tags):
    """Generate a hash for the cache key based on all parameters."""
    params = {
        "tags": sorted(tags) if tags else None,
    }
    params_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()


def _get_cache_path(cache_hash):
    """Get the cache file path for a given hash."""
    return CACHE_DIR / f"runs_{cache_hash}.pkl"


def _load_from_cache(cache_hash):
    """Load dataframe from cache if it exists."""
    cache_path = _get_cache_path(cache_hash)
    if cache_path.exists():
        try:
            logging.info(f"Loading runs from cache: {cache_path}")
            df = pd.read_pickle(cache_path)
            logging.info(f"Cache filepath: {cache_path.absolute()}")
            logging.info(f"Loaded {len(df)} runs from cache")
            return df
        except Exception as e:
            logging.warning(f"Failed to load cache: {e}")
            return None
    return None


def _save_to_cache(df, cache_hash):
    """Save dataframe to cache."""
    cache_path = _get_cache_path(cache_hash)
    try:
        df.to_pickle(cache_path)
        logging.info(f"Cached {len(df)} runs to: {cache_path}")
    except Exception as e:
        logging.warning(f"Failed to save cache: {e}")


def grid_status(pd_dataframe, map_hp_vals):
    lists = map_hp_vals.values()
    keys = map_hp_vals.keys()
    overview = dict()
    for element in itertools.product(*lists):
        instance = {}
        for i, k in enumerate(keys):
            if k not in pd_dataframe.columns:
                print(f"{k} not in cols {pd_dataframe.columns}")
                continue
            instance[k] = element[i]
        runs = pd_dataframe.loc[(pd_dataframe[list(instance)] == pd.Series(instance)).all(axis=1)]
        overview[str(instance)] = len(runs)

    return overview


#
# df1 = pd.DataFrame({'lr': [1, 1, 1, 3, 4], 'optim': ["sgd", "sgd", "sgd", "adam", "adam"]})
# cartesian_comp(df1, {"lr": [0, 1, 2, 3], "optim": ["sgd", "adam"]})


def diversity_check(pd_dataframe, hp_name, valeus_to_search):
    val_nums = [pd_dataframe[pd_dataframe[hp_name] == val] for val in valeus_to_search]

    return val_nums


# returns sorted metric values
def best_metric(pd_dataframe, metric_name, top_n=None, sort_invert=True):

    metric_list = []
    for index, row in pd_dataframe.iterrows():
        summ = row['summary']
        if metric_name in summ:
            acc = summ[metric_name]
            metric_list.append(acc)

    if top_n is not None and len(metric_list) < top_n:
        raise AssertionError(f"too little experiments {len(metric_list)} < {top_n} found: {metric_list}")

    if sort_invert:
        ret = sorted(metric_list)[::-1][:top_n]
    else:
        ret = sorted(metric_list)[:top_n]

    return ret[:top_n] if top_n is not None else ret


def filter_data(pf_datafframe, filter):
    return pf_datafframe.loc[(pf_datafframe[list(filter)] == pd.Series(filter)).all(axis=1)]


def _load_runs(tags=None, unfold=False):
    """
    Load runs from wandb filtered by tags only.
    Results are cached based on tags to speed up subsequent calls.
    
    Args:
        tags: List of tags to filter runs. If None or empty, all runs are loaded. Default: None
        unfold: Whether to unfold config and summary into separate columns. Default: False
    
    Returns:
        pd.DataFrame: DataFrame containing runs data with "best" column
    """
    # Check cache first
    cache_hash = _get_cache_hash(tags)
    cached_df = _load_from_cache(cache_hash)
    logging.info(f"Loading from cache with hash {cache_hash}")
    if cached_df is not None:
        return cached_df
    
    runs_all = 0

    api = wandb.Api()
    entity, project = WANDB_ENTITY, WANDB_PROJECT
    
    # Build filter for API to speed up queries
    filters = {}
    if tags:
        # Filter runs by tags in the API call
        filters["tags"] = {"$in": tags}
    
    runs = api.runs(entity + "/" + project, filters=filters)

    columns = ["summary", "config", "name", "best"]

    if not unfold:
        df = pd.DataFrame(columns=columns)
    else:
        df = None
        
    
    logging.info(f"Loading runs with tags {tags} from project {project} and entity {entity}") 

    for i_run, run in enumerate(runs):
        logging.info(f"Loaded {i_run} out of {len(runs)} runs with tags {tags}")
        logging.debug(f"RUN {run.name} with tags {run.tags}")
        runs_all += 1

        proto = {
            "summary": [run.summary._json_dict],
            "config": [
                {k: v for k, v in run.config.items()
                 if not k.startswith('_')}],
            "name": [run.name],
            "best": ["best" in run.tags]
        }

        if unfold:
            proto["name"] = proto["name"][0]
            for cfg_name, cfg_value in proto["config"][0].items():
                proto[f"config.{cfg_name}"] = cfg_value
            proto.pop("config")
            for summary_name, summary_value in proto["summary"][0].items():
                proto[f"summary.{summary_name}"] = summary_value
            proto.pop("summary")
            entry = pd.json_normalize(proto)
        else:
            entry = pd.DataFrame.from_dict(proto)

        if df is None:
            df = entry
        else:
            df = pd.concat([df, entry], ignore_index=True)

    
    # Save to cache
    _save_to_cache(df, cache_hash)

    return df


def load_runs(tags=None, mandatory_hp=None, mandatory_m=None, minimum_runtime_s=20, minimum_steps=500, unfold=False, best_only=False):
    """
    Load runs from wandb with filtering by tags and additional parameters.
    Results are cached based on all parameters to speed up subsequent calls.
    
    Args:
        tags: List of tags to filter runs. If None or empty, all runs are loaded. Default: None
        mandatory_hp: List of mandatory hyperparameters that must be present in run config. Default: None
        mandatory_m: List of mandatory metrics that must be present in run summary. Default: None
        minimum_runtime_s: Minimum runtime in seconds for a run. Default: 20
        minimum_steps: Minimum number of steps for a run. Default: 500
        unfold: Whether to unfold config and summary into separate columns. Default: False
    
    Returns:
        pd.DataFrame: DataFrame containing filtered runs data
    """
    # Check cache first
    cache_hash = _get_cache_hash_with_params(tags)
    cached_df = _load_from_cache(cache_hash)
    logging.info(f"Loading filtered runs from cache with hash {cache_hash}")
    if cached_df is not None:
        return cached_df
    
    # Load base runs with tags
    df_base = _load_runs(tags=tags, unfold=False)
    
    logging.info(f"Filtering {len(df_base)} runs with params - runtime>{minimum_runtime_s}s, steps>{minimum_steps}")
    
    runs_filtered = 0
    df_result = None
    
    for index, row in df_base.iterrows():
        js_summary = row['summary']
        if "_runtime" not in js_summary or "_step" not in js_summary:
            logging.debug(f"Skipping run {row['name']}: runtime or steps not present")
            continue
        if js_summary["_runtime"] < minimum_runtime_s:
            logging.debug(f"Skipping run {row['name']}: runtime {js_summary['_runtime']}s < {minimum_runtime_s}s")
            continue
        if js_summary["_step"] < minimum_steps:
            logging.debug(f"Skipping run {row['name']}: steps {js_summary['_step']} < {minimum_steps}")
            continue

        if best_only and not row['best']:
            logging.debug(f"Skipping run {row['name']}: not marked as best")
            continue
        
        ## mandatory M
        ok = True
        if mandatory_m:
            for m_m in mandatory_m:
                if m_m not in js_summary:
                    logging.debug(f"Skipping run {row['name']}: missing metric '{m_m}'")
                    ok = False
                    break
        if not ok:
            continue

        ## mandatory HP
        ok = True
        if mandatory_hp:
            js_config = row['config']
            for m_hp in mandatory_hp:
                if m_hp not in js_config:
                    logging.debug(f"Skipping run {row['name']}: missing hyperparameter '{m_hp}'")
                    ok = False
                    break
        if not ok:
            continue

        # Add mandatory metrics to row
        if mandatory_m:
            for m_m in mandatory_m:
                row[m_m] = js_summary[m_m]
        
        # Add mandatory hyperparameters to row
        if mandatory_hp:
            js_config = row['config']
            for m_hp in mandatory_hp:
                row[m_hp] = js_config[m_hp]
        
        # Add this row to results
        if df_result is None:
            df_result = pd.DataFrame([row])
        else:
            df_result = pd.concat([df_result, pd.DataFrame([row])], ignore_index=True)
        
        runs_filtered += 1

    if df_result is None:
        df_result = pd.DataFrame()
        logging.warning(f"No runs matched the filter criteria")
    else:
        logging.info(f"Filtered to {len(df_result)}/{len(df_base)} runs")
    
    # Handle unfold if requested
    if unfold:
        # Unfold the filtered results
        unfolded_rows = []
        for index, row in df_result.iterrows():
            proto = row.to_dict()
            name = proto["name"]
            for cfg_name, cfg_value in proto["config"].items():
                proto[f"config.{cfg_name}"] = cfg_value
            proto.pop("config")
            for summary_name, summary_value in proto["summary"].items():
                proto[f"summary.{summary_name}"] = summary_value
            proto.pop("summary")
            unfolded_rows.append(proto)
        df_result = pd.DataFrame(unfolded_rows)
    
    # Save to cache
    _save_to_cache(df_result, cache_hash)

    return df_result


def get_experiments(wandb_data: pd.DataFrame, **config):
    df = wandb_data.copy()
    for name, value in config.items():
        df = df.loc[df[name].isin([value])]
    return df


def has_experiment(wandb_data: pd.DataFrame, **config):
    df = wandb_data.copy()
    for name, value in config.items():
        df = df.loc[df[name].isin([value])]
    return len(df.index) > 0


def has_result_better_than(wandb_data: pd.DataFrame, result_limit: float, result_name: str, **config_filter):
    df = wandb_data.copy()
    for name, value in config_filter.items():
        df = df.loc[df[name].isin([value])]
    df = df.loc[df[result_name] > result_limit]
    return len(df.index) > 0


def has_result_less_than(wandb_data: pd.DataFrame, result_limit: float, result_name: str, **config_filter):
    df = wandb_data.copy()
    for name, value in config_filter.items():
        df = df.loc[df[name].isin([value])]
    df = df.loc[df[result_name] < result_limit]
    return len(df.index) > 0
