
import argparse
import torch
import pandas as pd
import wandb
import os

from training_utils.file_utils import open_json
from training_utils.model_utils import *
from training_utils.training import k_fold_training


config = open_json("simple_nn_config.json")
sweep_config = open_json("wandb_sweep_config.json")
DIR_PATH = config["dataset_dir"]
features_dict = open_json(f"{DIR_PATH}/features.json")
features, features_infos = compute_feature_list(config, features_dict)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--verbose", help="increase output verbosity", action="store_true")
parser.add_argument("--sweep_id", help="sweep_id to join an ongoing sweep")
args = parser.parse_args()

if args.sweep_id is None:
    sweep_id = wandb.sweep(sweep_config, project="nesp_simple_nn")
    print(f"set up new sweep {sweep_id=}")
else:
    sweep_id = args.sweep_id
    print(f"using given {sweep_id=}")


if args.verbose:
    print("verbosity turned on")
else:
    print("verbosity turned off")
    os.environ["WANDB_SILENT"] = "true"


if torch.cuda.is_available() and config["use_cuda"]:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(device)


# ## Wandb
# setup for hyper parameters optimization


def load_dataset(config):
    df = pd.read_csv(f"{DIR_PATH}/{config['dataset_name']}.csv")

    for feature in features:
        # remove line with nan values for selected features
        df = df[~(df[feature].isna())]

    # remove bad uniprot
    df = df[~(df["uniprot"].isin(config["bad_uniprot"]))]

    # apply max protein length
    df = df[df.length.le(config["max_protein_length"])]

    print(f"training on {len(df)} data")
    return df


# ## Wandb Sweep

global_config = config.copy()
main_df = load_dataset(global_config)


def run():
    # add protein_index to the dataset and get ksplit:
    df, ksplit = split_dataset(main_df, global_config)
    training_results, _, _ = k_fold_training(
        df, ksplit, global_config, features,
        features_infos, device, wandb_active=True,
        wandb_config=sweep_config)


try:
    wandb.agent(sweep_id, run, project="nesp_simple_nn")
    wandb.finish()
except Exception as e:
    print(f"Exception: {e}")
    wandb.finish()
