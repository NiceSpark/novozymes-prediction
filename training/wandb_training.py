#! /usr/bin/python3

"""
Python script for wandb sweep on hybrid nn or xgboost model
"""

import argparse
import wandb
import os

from training_utils.file_utils import open_json
from training_utils.model_utils import *
from training_utils.training_nn import k_fold_training_nn
from training_utils.training_xgboost import k_fold_training_xgboost

##### Arguments #####
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="nesp_hybrid_nn")
parser.add_argument("--sweep_id", help="sweep_id to join an ongoing sweep")
parser.add_argument("--xgboost", action="store_true")
parser.add_argument(
    "--verbose", help="increase output verbosity", action="store_true")
args = parser.parse_args()

##### Config #####
if args.xgboost:
    config = open_json("config_xgboost.json")
else:
    config = open_json("config_hybrid_nn.json")
    device = get_device(config)
    print(device)
    # if there is only one target replace "dTm_loss_coef" with either 1 (dTm in targets) or 0 (ddG in targets)
    if config["targets"] == ["ddG"]:
        config["dTm_loss_coef"] = 0
    elif config["targets"] == ["dTm"]:
        config["dTm_loss_coef"] = 1

sweep_config = open_json("wandb_sweep_config.json")
features_dict = open_json(
    f"{config['dataset_dir']}/{config['features_name']}.json")
features, features_infos = compute_feature_list(config, features_dict)

if args.sweep_id is None:
    sweep_id = wandb.sweep(sweep_config, project=args.name)
    print(f"set up new sweep {sweep_id=}")
else:
    sweep_id = args.sweep_id
    print(f"using given {sweep_id=}")


if args.verbose:
    print("verbosity turned on")
else:
    print("verbosity turned off")
    os.environ["WANDB_SILENT"] = "true"


##### Wandb Sweep #####

global_config = config.copy()
main_df = load_dataset(global_config, features, rm_nan=True)


def run():
    # add kfold to the dataset and get ksplit:
    df = split_dataset(main_df, global_config)
    _ = k_fold_training_nn(
        df, global_config, features,
        features_infos, device, wandb_active=True,
        wandb_config=sweep_config)
    if args.xgboost:
        print("training with xgboost")
        _ = k_fold_training_xgboost(df, global_config, features,
                                    features_infos,
                                    wandb_active=True,
                                    wandb_config=sweep_config)
    else:
        print("training with hybrid nn")
        _ = k_fold_training_nn(df, global_config, features,
                               features_infos, device, wandb_active=True,
                               wandb_config=sweep_config)


try:
    wandb.agent(sweep_id, run, project="nesp_hybrid_nn")
    wandb.finish()
except Exception as e:
    print(f"Exception: {e}")
    wandb.finish()
