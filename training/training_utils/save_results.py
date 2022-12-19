"""This module contains functions to save results and models."""

import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
from pickle import load
from glob import glob

from .file_utils import open_json, write_json


def move_models_and_scalers(dir_path: str):
    # we save models and scalers during training in tmp/,
    # we now need to move them to the appropriated folder
    for model_tmp_path in glob("tmp/model*.pth"):
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
    model_list, X_scaler_list = [], []
    ddG_scaler_list, dTm_scaler_list = [], []
    all_pca_directs = []

    for k in range(len(glob(f"{dir_path}/model_*.pth"))):
        model_list.append(torch.load(f"{dir_path}/model_{k}.pth"))
    for k in range(len(glob(f"{dir_path}/X_scaler_*.pkl"))):
        X_scaler_list.append(load(open(f"{dir_path}/X_scaler_{k}.pkl", 'rb')))
    for k in range(len(glob(f"{dir_path}/ddG_scaler_*.pkl"))):
        ddG_scaler_list.append(
            load(open(f"{dir_path}/ddG_scaler_{k}.pkl", 'rb')))
    for k in range(len(glob(f"{dir_path}/dTm_scaler_*.pkl"))):
        dTm_scaler_list.append(
            load(open(f"{dir_path}/dTm_scaler_{k}.pkl", 'rb')))
    for k in range(len(glob(f"{dir_path}/pca_direct_*.pkl"))):
        all_pca_directs.append(
            load(open(f"{dir_path}/pca_direct_{k}.pkl", 'rb')))

    all_scalers = {
        "X": X_scaler_list,
        "ddG": ddG_scaler_list,
        "dTm": dTm_scaler_list
    }

    return model_list, all_scalers, all_pca_directs


def save_submission(df, training_dir):
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


def plot_feature_weight(config, models, output_path, regression_only=False):
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


def log_kfold_training(name, results, config, features, model_structure):
    date_time = datetime.now()
    timestamp = date_time.strftime("%Y-%m-%d_%H-%M-%S")
    dir_num = len(glob(f"./outputs/{name}_*/"))
    dir_path = f"./outputs/{name}_{dir_num}"

    # add timestamp to all_training_results
    results = {"timestamp": timestamp, **results}
    config = {"timestamp": timestamp, **config}

    # create output subdir
    os.mkdir(dir_path)

    # log json infos
    write_json(f"{dir_path}/results.json", results)
    write_json(f"{dir_path}/config.json", config)
    write_json(f"{dir_path}/features.json", features)
    write_json(f"{dir_path}/model_structure.json", model_structure)

    # plot loss/mse over time for training on whole dataset
    simple_train_results = results["simple_train"]

    best_epoch_avg = sum(x.get("best_epoch", 0)
                         for x in simple_train_results)/int(config["kfold"])
    best_test_mse_avg = sum(x.get("best_test_mse")
                            for x in simple_train_results)/int(config["kfold"])
    training_time = sum(x.get('time', 0) for x in simple_train_results)
    loss_list = [x.get("loss_over_time") for x in simple_train_results]
    learning_rate_list = [x.get("learning_rate_over_time")
                          for x in simple_train_results]
    test_mse_list = [x.get("test_mse_over_time")
                     for x in simple_train_results]
    test_mse_ddG_list = [x.get("test_mse_ddG_over_time")
                         for x in simple_train_results]
    test_mse_dTm_list = [x.get("test_mse_dTm_over_time")
                         for x in simple_train_results]
    if (test_mse_list[0] is not None and loss_list[0] is not None):
        colors = ['r', 'c', 'b', 'm', 'y', 'r--', 'c--', 'b--', 'm--', 'y--']

        plt.figure(figsize=(20, 8))
        plt.title(
            f"Training Results on {config['dataset_name']} for {name}_{dir_num}")
        plt.suptitle(
            f"{best_test_mse_avg= :.2f}, {best_epoch_avg= :.2f}, {training_time= :.2f}")

        # plt.subplot(1, 3, 1)
        # plt.title("learning rate over time")
        # for i, lr in enumerate(learning_rate_list):
        #     plt.plot(lr, colors[i], label=f"learning_rate_{i}")

        plt.subplot(1, 4, 1)
        plt.title("loss over time")
        for i, loss in enumerate(loss_list):
            plt.plot(loss, colors[i], label=f"loss_{i}")

        plt.subplot(1, 4, 2)
        plt.title("test_mse_ddG_over_time")
        for i, test_mse_ddG in enumerate(test_mse_ddG_list):
            plt.plot(test_mse_ddG, colors[i], label=f"test_mse_ddG_{i}")

        plt.subplot(1, 4, 3)
        plt.title("test_mse_dTm_over_time")
        for i, test_mse_dTm in enumerate(test_mse_dTm_list):
            plt.plot(test_mse_dTm, colors[i], label=f"test_mse_dTm_{i}")

        plt.subplot(1, 4, 4)
        plt.title("test mse over time")
        for i, test_mse in enumerate(test_mse_list):
            plt.plot(test_mse, colors[i], label=f"test_mse_{i}")

        plt.savefig(f"{dir_path}/results.jpg")

    # plot feature importance
    models, _, _ = load_models_and_scalers("tmp")
    plot_feature_weight(config, models, f"{dir_path}/features_importance.png", regression_only=(
        "regression_only" in config["model_type"]))
    return dir_path


def log_learning_curve(name, all_training_results, config, features, model_structure):
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
    write_json(f"{dir_path}/model_structure.json", model_structure)

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
    plt.savefig(f"{dir_path}/learning_curve.jpg")

    # plot loss/mse over time for training on whole dataset
    simple_train_results = all_training_results["training_by_num_rows"][-1]
    train_avg_mse = sum(x['train_mse']
                        for x in simple_train_results)/config["kfold"]
    test_avg_mse = sum(x['test_mse']
                       for x in simple_train_results)/config["kfold"]
    training_time = sum(x.get('time', 0) for x in simple_train_results)
    loss_list = [x.get("loss_over_time") for x in simple_train_results]
    train_mse_list = [x.get("train_mse_over_time")
                      for x in simple_train_results]
    test_mse_list = [x.get("test_mse_over_time")
                     for x in simple_train_results]

    if (test_mse_list[0] is not None and loss_list[0] is not None):
        plt.figure(figsize=(10, 5))
        plt.title(
            f"Training Results on {config['dataset_name']} for {name}_{dir_num}")
        plt.suptitle(
            f"{train_avg_mse= :.2f}, {test_avg_mse= :.2f}, {training_time= :.2f}")
        plt.subplot(1, 2, 1)
        plt.title("loss over time")
        for loss in loss_list:
            plt.plot(loss)

        plt.subplot(1, 2, 2)
        plt.title("mse over time")
        for train_mse in train_mse_list:
            plt.plot(train_mse, "r", label="train_mse")
        for test_mse in test_mse_list:
            plt.plot(test_mse, "g", label="test_mse")
        plt.savefig(f"{dir_path}/results.jpg")

    return dir_path
