import copy
import torch.nn as nn
import torch
import time
import tqdm
import gc
import wandb
from torch.optim import AdamW
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
from pickle import dump
from .models import HybridNN
from .model_utils import *


def train_model(model: nn.Module, config: dict,
                optimizer, loss_function, trainloader, device,
                X_test_voxel, X_test_features, y_test):

    loss_over_time = []
    train_mse_over_time = []
    test_mse_over_time = []

    # Run the training loop
    for epoch in range(config["num_epochs"]):

        # set model in train mode
        model.train()
        # Set current loss value
        current_loss = 0.0
        current_mse = 0
        i = 0
        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):
            # Get and prepare inputs
            voxel_inputs, features_inputs, targets = data
            # inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))
            # Move to cuda device
            voxel_inputs = voxel_inputs.to(device)
            features_inputs = features_inputs.to(device)
            targets = targets.to(device)
            # Zero the gradients
            optimizer.zero_grad()
            # Perform forward pass
            outputs = model(voxel_inputs, features_inputs)
            # Compute loss
            loss = loss_function(outputs, targets)
            # Perform backward pass
            loss.backward()
            # Perform optimization
            optimizer.step()

            # compute statistics
            current_mse += mean_squared_error(
                targets.cpu().detach().numpy(), outputs.cpu().detach().numpy())
            current_loss += loss.item()

        loss_over_time.append(current_loss/(i+1))
        train_mse_over_time.append(current_mse/(i+1))

        model.eval()
        with torch.set_grad_enabled(False):
            test_mse, _ = evaluate_model(
                X_test_voxel, X_test_features, y_test, model, device)
            test_mse_over_time.append(test_mse)

    return model, loss_over_time, train_mse_over_time, test_mse_over_time


def build_optimizer(model, config):
    if config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=config["learning_rate"], momentum=0.9)
    elif config["optimizer"] == "adamw":
        def get_optimizer_config(model, encoder_lr, weight_decay=0.0):
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'lr': encoder_lr, 'weight_decay': weight_decay},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                 'lr': encoder_lr, 'weight_decay': 0.0},
            ]
            return optimizer_parameters

        optimizer_parameters = get_optimizer_config(model,
                                                    encoder_lr=config['learning_rate'],
                                                    weight_decay=config['AdamW_decay'])
        optimizer = AdamW(optimizer_parameters, lr=config['learning_rate'])
    else:  # config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config["learning_rate"])
    return optimizer


def k_fold_training(df, global_config, features, features_infos,
                    device, wandb_active=False, wandb_config={}, keep_models=False):
    training_results = []
    model_list = [None]*global_config["kfold"]
    scaler_list = [None]*global_config["kfold"]

    if wandb_active:
        wandb.init(config=wandb_config)
        config = wandb.config
    else:
        config = global_config

    used_torch_type = torch.double if config["use_double"] else torch.float

    for k in tqdm.tqdm(range(global_config["kfold"])):
        train = list(range(global_config["kfold"]))
        test = [train.pop(k)]
        df_train = df[df["kfold"].isin(train)]
        df_test = df[df["kfold"].isin(test)]

        # we load the data for training
        dataset_train = prepare_train_data(
            df_train, config, features, features_infos)
        trainloader = DataLoader(dataset_train,
                                 batch_size=config["batch_size"],
                                 shuffle=config["shuffle_dataloader"],
                                 num_workers=global_config["num_workers"])

        # we load the data for evaluation
        X_train_voxel, X_train_features, y_train = prepare_eval_data(
            df_train, config, features, features_infos, dataset_train.scaler)
        X_test_voxel, X_test_features, y_test = prepare_eval_data(
            df_test, config, features, features_infos, dataset_train.scaler)

        # Initialize a new Novozymes Model
        model = HybridNN(
            len(features_infos["direct_features"]), config)
        model.to(used_torch_type)
        model.to(device)

        # Define the loss function and optimizer
        loss_function = nn.MSELoss()
        optimizer = build_optimizer(model, config)
        # Train model:
        t0 = time.time()
        model, loss_over_time, train_mse_over_time, test_mse_over_time = train_model(
            model, config, optimizer, loss_function,
            trainloader, device, X_test_voxel, X_test_features, y_test)

        t1 = time.time()-t0

        # Evaluate this model:
        model.eval()
        with torch.set_grad_enabled(False):
            train_mse, _ = evaluate_model(
                X_train_voxel, X_train_features, y_train, model, device)
            test_mse, test_diff = evaluate_model(
                X_test_voxel, X_test_features, y_test, model, device)
            # get worst samples
            worst_samples = get_worst_samples(
                df_test, test_diff, global_config)

            # print(f"MSE obtained for k-fold {k}: {mse}")

            results = {
                "loss_over_time": loss_over_time,
                "train_mse_over_time": train_mse_over_time,
                "test_mse_over_time": test_mse_over_time,
                "worst_samples": worst_samples,
                "train_mse": train_mse,
                "test_mse": test_mse,
                "time": t1
            }
            training_results.append(results)

        if keep_models:
            torch.save(model, f"tmp/model_{k}.pth")
            dump(dataset_train.scaler, open(f"tmp/scaler_{k}.pkl", "wb"))

        # end of process for k, freeing memory
        del model
        del df_train, df_test
        del dataset_train, trainloader
        del X_train_voxel, X_train_features, y_train
        del X_test_voxel, X_test_features, y_test
        gc.collect()
        torch.cuda.empty_cache()

    # Process is complete.
    if wandb_active:
        num_epochs = config["num_epochs"]
        avg_test_mse_over_time = np.mean(
            np.array([r["test_mse_over_time"] for r in training_results]), axis=0)
        avg_train_mse_over_time = np.mean(
            np.array([r["train_mse_over_time"] for r in training_results]), axis=0)
        avg_loss_over_time = np.mean(
            np.array([r["loss_over_time"] for r in training_results]), axis=0)

        for epoch in range(num_epochs):
            wandb.log({
                "test_mse": avg_test_mse_over_time[epoch],
                "train_mse": avg_train_mse_over_time[epoch],
                "loss": avg_loss_over_time[epoch],
            })

    if wandb_active:
        wandb.finish()

    return training_results
