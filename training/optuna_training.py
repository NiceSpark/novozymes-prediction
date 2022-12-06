
import argparse
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback

from training_utils.file_utils import open_json
from training_utils.model_utils import *
from training_utils.training import k_fold_training

N_OPTUNA_TRIALS = 40

config = open_json("hybrid_nn_config.json")
features_dict = open_json(
    f"{config['dataset_dir']}/{config['features_name']}.json")
features, features_infos = compute_feature_list(config, features_dict)
device = get_device(config)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--verbose", help="increase output verbosity", action="store_true")
args = parser.parse_args()


if args.verbose:
    print("verbosity turned on")

global_config = config.copy()
main_df = load_dataset(global_config, features, rm_nan=True)


wandbc = WeightsAndBiasesCallback(wandb_kwargs={"project": "hybrid_nn_optuna_2"},
                                  as_multirun=True)


@wandbc.track_in_wandb()
def objective(trial):
    optuna_config = global_config.copy()
    optuna_config["regression_fc_layer_size"] = trial.suggest_int(
        "regression_fc_layer_size", 512, 1024, log=True)
    optuna_config["regression_hidden_layers"] = trial.suggest_int(
        "regression_hidden_layers", 1, 7)
    optuna_config["regression_dropout"] = trial.suggest_float(
        "regression_dropout", 0., 0.3, log=False)

    print("optuna_config:", optuna_config)

    # --------------- train --------------

    # add protein_index to the dataset and get ksplit:
    df = split_dataset(main_df, global_config)
    # train
    training_results = k_fold_training(
        df, global_config, features,
        features_infos, device, wandb_active=False)
    # compute avg_mse
    test_mse = sum(x["test_mse"]
                   for x in training_results)/config["kfold"]
    return test_mse


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=N_OPTUNA_TRIALS, callbacks=[wandbc])
