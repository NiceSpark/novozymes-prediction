"""This file contains helper functions for writing json documents, as well as creating timestamp (that you need for those json)"""

import json
import glob
import os
import matplotlib.pyplot as plt
from datetime import datetime


def open_json(filename):
    """
    open a .json file from the file path (filename)
    """
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    else:
        print(f"file {filename} does not exists !")
        return {}


def write_json(filename, json_file):
    """
    write a json_file to the file path (filename)
    """
    with open(filename, "w+") as f:
        json.dump(json_file, f, indent=4, default=str)


def create_timestamp():
    dateTimeObj = datetime.now()
    return dateTimeObj


def save_submission(df, name="submission"):
    date_time = datetime.now()
    timestamp = date_time.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = f"./submissions/{name}_{timestamp}.csv"
    df.to_csv(save_path, index=False)
    return save_path


def log_kfold_training(name, results, config, features, model_structure):
    date_time = datetime.now()
    timestamp = date_time.strftime("%Y-%m-%d_%H-%M-%S")
    dir_num = len(glob.glob(f"./outputs/{name}_*/"))
    dir_path = f"./outputs/{name}_{dir_num}"

    # add timestamp to all_training_results
    results["timestamp"] = timestamp

    # create output subdir
    os.mkdir(dir_path)

    # log json infos
    write_json(f"{dir_path}/results.json", results)
    write_json(f"{dir_path}/config.json", config)
    write_json(f"{dir_path}/features.json", features)
    write_json(f"{dir_path}/model_structure.json", model_structure)

    # plot loss/mse over time for training on whole dataset
    simple_train_results = results["simple_train"]

    best_epoch_avg = sum(x['best_epoch']
                         for x in simple_train_results)/config["kfold"]
    best_test_mse_avg = sum(x['best_test_mse']
                            for x in simple_train_results)/config["kfold"]
    training_time = sum(x.get('time', 0) for x in simple_train_results)
    loss_list = [x.get("loss_over_time") for x in simple_train_results]
    learning_rate_list = [x.get("learning_rate_over_time")
                          for x in simple_train_results]
    train_mse_list = [x.get("train_mse_over_time")
                      for x in simple_train_results]
    test_mse_list = [x.get("test_mse_over_time")
                     for x in simple_train_results]

    if (test_mse_list[0] is not None and loss_list[0] is not None):
        colors = ['r', 'c', 'b', 'm', 'y']

        plt.figure(figsize=(10, 5))
        plt.title(
            f"Training Results on {config['dataset_name']} for {name}_{dir_num}")
        plt.suptitle(
            f"{best_test_mse_avg= :.2f}, {best_epoch_avg= :.2f}, {training_time= :.2f}")

        plt.subplot(1, 3, 1)
        plt.title("learning rate over time")
        for i, lr in enumerate(learning_rate_list):
            plt.plot(lr, colors[i], label=f"learning_rate_{i}")

        plt.subplot(1, 3, 2)
        plt.title("train mse over time")
        for i, train_mse in enumerate(train_mse_list):
            plt.plot(train_mse, colors[i], label=f"train_mse_{i}")

        plt.subplot(1, 3, 3)
        plt.title("test mse over time")
        for i, test_mse in enumerate(test_mse_list):
            plt.plot(test_mse, colors[i], label=f"test_mse_{i}")

        plt.savefig(f"{dir_path}/results.jpg")

    return dir_path


def log_learning_curve(name, all_training_results, config, features, model_structure):
    date_time = datetime.now()
    timestamp = date_time.strftime("%Y-%m-%d_%H-%M-%S")
    dir_num = len(glob.glob(f"./outputs/{name}_*"))
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
