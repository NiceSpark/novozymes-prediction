#! /usr/bin/python3

"""
Python script for training a hybrid neural network or xgboost model
"""

import argparse

from training_utils.file_utils import open_json
from training_utils.save_results import log_kfold_training, move_models_and_scalers
from training_utils.models import HybridNN
from training_utils.model_utils import *
from training_utils.training_nn import k_fold_training_nn
from training_utils.training_xgboost import k_fold_training_xgboost

# use argparse to get config: xgboost or hybrid, keep_models
parser = argparse.ArgumentParser()
# add argument xgboost if you want to use xgboost
parser.add_argument("--xgboost", action="store_true")
parser.add_argument("--keep_models", action="store_true")
args = parser.parse_args()

if args.xgboost:
    config = open_json("config_xgboost.json")
else:
    config = open_json("config_hybrid_nn.json")
    device = get_device(config)
    # if there is only one target replace "dTm_loss_coef" with either 1 (dTm in targets) or 0 (ddG in targets)
    if config["targets"] == ["ddG"]:
        config["dTm_loss_coef"] = 0
    elif config["targets"] == ["dTm"]:
        config["dTm_loss_coef"] = 1

features_dict = open_json(
    f"{config['dataset_dir']}/{config['features_name']}.json")
features, features_infos = compute_feature_list(config, features_dict)
log_name = config["model_type"]

print(f"using {len(features_infos['direct_features'])} direct features")

df = load_dataset(config, features, rm_nan=True)
# add kfold group to the dataset and get ksplit:
df = split_dataset(df, config)

# training
if args.xgboost:
    print("training with xgboost")
    training_results = k_fold_training_xgboost(
        df, config, features, features_infos, keep_models=args.keep_models)
    model_structure = []
else:
    print("training with hybrid nn")
    training_results = k_fold_training_nn(
        df, config, features, features_infos, device, keep_models=args.keep_models)
    # save results to output
    model = HybridNN(config)
    model_structure = str(model).replace(
        '(', '').replace(')', '').split('\n')

# save results to output
dir_path = log_kfold_training(
    log_name, training_results, config, features, model_structure)

# move models and scalers to output
if args.keep_models:
    move_models_and_scalers(dir_path)

print(f"logged training in {dir_path}")
