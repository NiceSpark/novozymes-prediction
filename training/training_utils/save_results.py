"""
This module contains functions to save results
but also models and scalers in order to predict on submission dataset at a latter time.
"""

import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import torch
import xgboost as xg
from datetime import datetime
from pickle import load
from glob import glob

from .file_utils import open_json, write_json


def move_models_and_scalers(dir_path: str):
    """
    we save models and scalers during training in tmp/,
    we now need to move them to the appropriated folder
    """
    for model_tmp_path in glob("tmp/model*"):
        shutil.move(model_tmp_path, f"{dir_path}/")
    for scaler_tmp_path in glob("tmp/X_scaler*.pkl"):
        shutil.move(scaler_tmp_path, f"{dir_path}/")
    for scaler_tmp_path in glob("tmp/dTm_scaler*.pkl"):
        shutil.move(scaler_tmp_path, f"{dir_path}/")
    for scaler_tmp_path in glob("tmp/ddG_scaler*.pkl"):
        shutil.move(scaler_tmp_path, f"{dir_path}/")
    for pca_tmp_path in glob("tmp/pca*.pkl"):
        shutil.move(pca_tmp_path, f"{dir_path}/")


def load_models_and_scalers(dir_path: str):
    """
    load models and scalers from dir_path, in order to predict on submission dataset
    """
    model_list, X_scaler_list = [], []
    ddG_scaler_list, dTm_scaler_list = [], []
    all_pca_directs = []

    if "xgboost" in dir_path:
        # load xgboost model from json file
        for k in range(len(glob(f"{dir_path}/model_*.json"))):
            model = xg.XGBRegressor()
            model.load_model(f"{dir_path}/model_{k}.json")
            model_list.append(model)
    else:
        # load pytorch model from pth file
        for k in range(len(glob(f"{dir_path}/model_*.pth"))):
            model_list.append(torch.load(f"{dir_path}/model_{k}.pth"))
    # load X scaler
    for k in range(len(glob(f"{dir_path}/X_scaler_*.pkl"))):
        X_scaler_list.append(load(open(f"{dir_path}/X_scaler_{k}.pkl", 'rb')))
    # load ddG scaler
    for k in range(len(glob(f"{dir_path}/ddG_scaler_*.pkl"))):
        ddG_scaler_list.append(
            load(open(f"{dir_path}/ddG_scaler_{k}.pkl", 'rb')))
    # load dTm scaler
    for k in range(len(glob(f"{dir_path}/dTm_scaler_*.pkl"))):
        dTm_scaler_list.append(
            load(open(f"{dir_path}/dTm_scaler_{k}.pkl", 'rb')))
    # load pca
    for k in range(len(glob(f"{dir_path}/pca_direct_*.pkl"))):
        all_pca_directs.append(
            load(open(f"{dir_path}/pca_direct_{k}.pkl", 'rb')))

    all_scalers = {
        "X": X_scaler_list,
        "ddG": ddG_scaler_list,
        "dTm": dTm_scaler_list
    }

    return model_list, all_scalers, all_pca_directs


def save_submission(df, training_dir: str):
    """
    save submission csv file in submissions/ folder
    also include in the folder the config and result json as well as the training plot
    """
    date_time = datetime.now()
    timestamp = date_time.strftime("%Y-%m-%d_%H-%M-%S")
    training_config = open_json(training_dir+"/config.json")
    name = training_config["model_type"]
    dir_path = f"./submissions/{name}_{timestamp}"
    save_path = f"{dir_path}/{name}_{timestamp}.csv"

    # create dir
    os.mkdir(dir_path)
    # save submission csv
    df.to_csv(save_path, index=False)
    # save config infos and results for reference
    for path in glob(f"{training_dir}/*.json")+glob(f"{training_dir}/*.jpg"):
        shutil.copy(path, dir_path)

    return save_path


def compute_feature_weight(model):
    """
    compute feature weight for each model
    """
    weights = []
    for name, params in model.named_parameters():
        if "regression_model" in name and "weight" in name:
            weights.append(params.cpu().detach().numpy())

    summed_weights = np.array([1])
    for params in reversed(weights):
        params = np.multiply(summed_weights, params)
        summed_weights = abs(params).sum(axis=0)
        summed_weights = summed_weights.reshape(summed_weights.shape[0], 1)

    return summed_weights


def plot_feature_weight(config: dict, models: list,
                        output_path: str, regression_only=False):
    """
    plot feature weight for each model and save it to the output_path
    """

    _ = plt.figure(figsize=(15, 5))
    all_summed_weights = []
    for model in models:
        summed_weights = compute_feature_weight(model)
        summed_weights = summed_weights/np.linalg.norm(summed_weights)
        all_summed_weights.append(summed_weights)
        plt.plot(summed_weights, '.', markersize=5)

    avg_summed_weights = np.array(all_summed_weights).mean(axis=0)
    plt.title("Feature importance")
    plt.plot(avg_summed_weights, 'k.', markersize=7)
    if not regression_only:
        plt.axvline(config["cnn_dense_layer_size"] -
                    0.5, color="k", linestyle="--")
    plt.savefig(output_path)


def plot_xgboost_training(results: dict, config: dict,
                          name: str, dir_path: str, dir_num: int,
                          colors: list):
    """
    plot training results for xgboost
    """
    # get data
    avg_train_mse = sum(x.get("train_mse", 0)
                        for x in results)/int(config["kfold"])
    avg_test_mse = sum(x.get("test_mse", 0)
                       for x in results)/int(config["kfold"])

    # plot results for xgboost
    plt.figure(figsize=(10, 4))
    plt.title(
        f"Training Results on {config['dataset_name']} for {name}_{dir_num}")
    plt.suptitle(
        f"{avg_train_mse= :.2f}, {avg_test_mse= :.2f}")
    plt.subplot(1, 2, 1)
    plt.title("train mse")
    for i, train_mse in enumerate([x.get("train_mse") for x in results]):
        plt.plot(train_mse, colors[i %
                                   len(colors)]+'o', label=f"train_mse_{i}")
    plt.subplot(1, 2, 2)
    plt.title("test mse")
    for i, test_mse in enumerate([x.get("test_mse") for x in results]):
        plt.plot(test_mse, colors[i %
                                  len(colors)]+'o', label=f"test_mse_{i}")
    # save figure to dir_path
    plt.savefig(f"{dir_path}/results.jpg")


def plot_nn_training(results: dict, config: dict,
                     name: str, dir_path: str, dir_num: int,
                     colors: list):
    """
    plot training results for neural network
    """
    # get data
    best_epoch_avg = sum(x.get("best_epoch", 0)
                         for x in results)/int(config["kfold"])
    best_test_mse_avg = sum(x.get("best_test_mse", 0)
                            for x in results)/int(config["kfold"])
    training_time = sum(x.get('time', 0) for x in results)
    loss_list = [x.get("loss_over_time", 0) for x in results]
    test_mse_list = [x.get("test_mse_over_time", 0) for x in results]
    test_mse_ddG_list = [x.get("test_mse_ddG_over_time", 0)
                         for x in results]
    test_mse_dTm_list = [x.get("test_mse_dTm_over_time", 0)
                         for x in results]
    # plot the loss, mse, mse_ddG, mse_dTm over time
    plt.figure(figsize=(20, 8))
    plt.title(
        f"Training Results on {config['dataset_name']} for {name}_{dir_num}")
    plt.suptitle(
        f"{best_test_mse_avg= :.2f}, {best_epoch_avg= :.2f}, {training_time= :.2f}")
    plt.subplot(1, 4, 1)
    plt.title("loss over time")
    for i, loss in enumerate(loss_list):
        plt.plot(loss, colors[i % len(colors)], label=f"loss_{i}")
    plt.subplot(1, 4, 2)
    plt.title("test_mse_ddG_over_time")
    for i, test_mse_ddG in enumerate(test_mse_ddG_list):
        plt.plot(test_mse_ddG, colors[i % len(
            colors)], label=f"test_mse_ddG_{i}")
    plt.subplot(1, 4, 3)
    plt.title("test_mse_dTm_over_time")
    for i, test_mse_dTm in enumerate(test_mse_dTm_list):
        plt.plot(test_mse_dTm, colors[i % len(
            colors)], label=f"test_mse_dTm_{i}")
    plt.subplot(1, 4, 4)
    plt.title("test mse over time")
    for i, test_mse in enumerate(test_mse_list):
        plt.plot(test_mse, colors[i % len(colors)], label=f"test_mse_{i}")

    # save figure to dir_path
    plt.savefig(f"{dir_path}/results.jpg")

    # plot feature importance
    models, _, _ = load_models_and_scalers("tmp")
    plot_feature_weight(config, models, f"{dir_path}/features_importance.png", regression_only=(
        "regression_only" in config["model_type"]))


def log_kfold_training(name: str, results: dict, config: dict,
                       features: list, model_structure: str):
    """
    log training results for kfold training
    works for both nn and xgboost
    """

    date_time = datetime.now()
    timestamp = date_time.strftime("%Y-%m-%d_%H-%M-%S")
    dir_num = len(glob(f"./outputs/{name}_*/"))
    dir_path = f"./outputs/{name}_{dir_num}"
    colors = ['r', 'c', 'b', 'm', 'y', 'r--', 'c--', 'b--', 'm--', 'y--']

    # create output subdir
    os.mkdir(dir_path)

    # log json infos
    write_json(f"{dir_path}/results.json",
               {"timestamp": timestamp, "results": results})
    write_json(f"{dir_path}/config.json", {"timestamp": timestamp, **config})
    write_json(f"{dir_path}/features.json", features)
    if model_structure:
        write_json(f"{dir_path}/model_structure.json", model_structure)

    # plot loss/mse over time for training on whole dataset
    if config["model_type"] != "xgboost":
        plot_nn_training(results, config, name, dir_path, dir_num, colors)
    else:
        plot_xgboost_training(results, config, name, dir_path, dir_num, colors)

    return dir_path


def log_learning_curve(name: str, all_training_results: dict,
                       config: dict, features: list):
    """
    log training results for learning curve computation
    works for both nn and xgboost
    """

    date_time = datetime.now()
    timestamp = date_time.strftime("%Y-%m-%d_%H-%M-%S")
    dir_num = len(glob(f"./outputs/{name}_*"))
    dir_path = f"./outputs/{name}_{dir_num}"

    # add timestamp to all_training_results
    all_training_results["timestamp"] = timestamp

    # create output subdir
    os.mkdir(dir_path)
    # log json infos
    write_json(f"{dir_path}/all_training_results.json", all_training_results)
    write_json(f"{dir_path}/config.json", config)
    write_json(f"{dir_path}/features.json", features)

    # create the different plots
    num_rows = all_training_results["learning_curve"]["num_rows"]
    train_mse = all_training_results["learning_curve"]["train_mse"]
    test_mse = all_training_results["learning_curve"]["test_mse"]

    # learning curve
    plt.figure()
    plt.title(f"Learning Curve for {name}_{dir_num}")
    plt.plot(num_rows, train_mse, "r", label="train_mse")
    plt.plot(num_rows, test_mse, "g", label="test_mse")
    plt.legend()
    plt.savefig(f"{dir_path}/learning_curve.png")

    return dir_path
