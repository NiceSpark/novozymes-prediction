#! /usr/bin/python3

"""
Python script for hyperparameter optimization using optuna
"""

import argparse
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
from pprint import pprint

from training_utils.file_utils import open_json
from training_utils.model_utils import *
from training_utils.training_nn import k_fold_training_nn
from training_utils.training_xgboost import k_fold_training_xgboost

##### Arguments #####
parser = argparse.ArgumentParser()
parser.add_argument("--n_trials", type=int, default=10)
parser.add_argument("--name", type=str, default="optuna_study")
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
    # if there is only one target replace "dTm_loss_coef" with either 1 (dTm in targets) or 0 (ddG in targets)
    if config["targets"] == ["ddG"]:
        config["dTm_loss_coef"] = 0
    elif config["targets"] == ["dTm"]:
        config["dTm_loss_coef"] = 1

features_dict = open_json(
    f"{config['dataset_dir']}/{config['features_name']}.json")
features, features_infos = compute_feature_list(config, features_dict)


if args.verbose:
    print("verbosity turned on")

global_config = config.copy()
main_df = load_dataset(global_config, features, rm_nan=True)

# set up callback for wandb logging and visualization
wandbc = WeightsAndBiasesCallback(wandb_kwargs={"project": args.name},
                                  as_multirun=True)


@wandbc.track_in_wandb()
def objective(trial):
    """
    objective function for optuna study
    """
    optuna_config = global_config.copy()
    if args.xgboost:
        optuna_config["max_depth"] = trial.suggest_int("max_depth", 5, 12)
        optuna_config["learning_rate"] = trial.suggest_float(
            "learning_rate", 1e-3, 1, log=True)
        optuna_config["n_estimators"] = trial.suggest_int(
            "n_estimators", 250, 1500)
        pprint({f"{k}: {optuna_config[k]}" for k in ["max_depth",
                                                     "learning_rate",
                                                     "n_estimators"]})

    else:
        optuna_config["regression_fc_layer_size"] = trial.suggest_int(
            "regression_fc_layer_size", 512, 1024, log=True)
        optuna_config["regression_hidden_layers"] = trial.suggest_int(
            "regression_hidden_layers", 1, 7)
        optuna_config["regression_dropout"] = trial.suggest_float(
            "regression_dropout", 0., 0.3, log=False)
        # print optuna_config for this run
        pprint({f"{k}: {optuna_config[k]}" for k in ["regression_fc_layer_size",
                                                     "regression_hidden_layers",
                                                     "regression_dropout"]})

    # add kfold group to the dataset and get ksplit:
    df = split_dataset(main_df, config)
    # training
    if args.xgboost:
        print("training with xgboost")
        training_results = k_fold_training_xgboost(
            df, optuna_config, features, features_infos)
    else:
        print("training with hybrid nn")
        training_results = k_fold_training_nn(
            df, optuna_config, features, features_infos, device)

    # compute avg_mse
    test_mse = sum(x["test_mse"]
                   for x in training_results)/config["kfold"]
    return test_mse


# run optuna study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=args.n_trials, callbacks=[wandbc])
