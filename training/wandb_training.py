
import argparse
import torch
import pandas as pd
import wandb
import os

from training_utils.file_utils import open_json
from training_utils.model_utils import *
from training_utils.training_nn import k_fold_training_nn


config = open_json("config_hybrid_nn.json")
sweep_config = open_json("wandb_sweep_config.json")
features_dict = open_json(
    f"{config['dataset_dir']}/{config['features_name']}.json")
features, features_infos = compute_feature_list(config, features_dict)
device = get_device(config)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--verbose", help="increase output verbosity", action="store_true")
parser.add_argument("--sweep_id", help="sweep_id to join an ongoing sweep")
args = parser.parse_args()

if args.sweep_id is None:
    sweep_id = wandb.sweep(sweep_config, project="nesp_hybrid_nn")
    print(f"set up new sweep {sweep_id=}")
else:
    sweep_id = args.sweep_id
    print(f"using given {sweep_id=}")


if args.verbose:
    print("verbosity turned on")
else:
    print("verbosity turned off")
    os.environ["WANDB_SILENT"] = "true"

print(device)

# ## Wandb Sweep

global_config = config.copy()
main_df = load_dataset(global_config, features, rm_nan=True)


def run():
    # add protein_index to the dataset and get ksplit:
    df = split_dataset(main_df, global_config)
    _ = k_fold_training_nn(
        df, global_config, features,
        features_infos, device, wandb_active=True,
        wandb_config=sweep_config)


try:
    wandb.agent(sweep_id, run, project="nesp_hybrid_nn")
    wandb.finish()
except Exception as e:
    print(f"Exception: {e}")
    wandb.finish()
