import copy
import torch.nn as nn
import torch
import time
import tqdm
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

from .models import SimpleNN
from .model_utils import *


def train_model(model: SimpleNN, config, optimizer, loss_function,
                trainloader, device,
                X_test=None, y_test=None):

    loss_over_time = []
    train_mse_over_time = []
    test_mse_over_time = []

    # Run the training loop
    for epoch in tqdm.tqdm(range(config["num_epochs"])):
        # set model in train mode
        model.train()
        # Set current loss value
        current_loss = 0.0
        current_mse = 0
        i = 0
        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):
            # Get and prepare inputs
            inputs, targets = data
            # inputs, targets = inputs.float(), targets.float()
            targets = targets.reshape((targets.shape[0], 1))
            # Move to cuda device
            inputs, targets = inputs.to(device), targets.to(device)
            # Zero the gradients
            optimizer.zero_grad()
            # Perform forward pass
            outputs = model(inputs)
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

        if X_test is not None:
            model.eval()
            with torch.set_grad_enabled(False):
                test_mse, _ = evaluate_model(X_test, y_test, model, device)
                test_mse_over_time.append(test_mse)

    return model, loss_over_time, train_mse_over_time, test_mse_over_time


def k_fold_training(df, ksplit, config, features, features_infos, device, keep_models=False):
    training_results = []
    model_list = [None]*config["k-fold"]
    scaler_list = [None]*config["k-fold"]

    for k in range(config["k-fold"]):
        train, test = next(ksplit)
        df_train = df[df["protein_index"].isin(train)]
        df_test = df[df["protein_index"].isin(test)]

        # we load the data for training
        dataset_train = prepare_train_data(
            df_train, config, features, features_infos)
        trainloader = DataLoader(dataset_train,
                                 batch_size=config["batch_size"],
                                 shuffle=config["shuffle_dataloader"],
                                 num_workers=config["num_workers"])

        # we load the data for evaluation
        X_train, y_train = prepare_eval_data(
            df_train, config, features, features_infos, dataset_train.scaler)
        X_test, y_test = prepare_eval_data(
            df_test, config, features, features_infos, dataset_train.scaler)

        # Initialize a new Novozymes Model
        model = SimpleNN(
            len(features_infos["direct_features"]), config["model_config"])
        model.to(torch.double)
        model.to(device)

        # Define the loss function and optimizer
        loss_function = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Train model:
        t0 = time.time()
        model, loss_over_time, train_mse_over_time, test_mse_over_time = train_model(
            model, config, optimizer, loss_function, trainloader, device, X_test, y_test)
        t1 = time.time()-t0

        # Evaluate this model:
        model.eval()
        with torch.set_grad_enabled(False):
            train_mse, _ = evaluate_model(X_train, y_train, model, device)
            test_mse, test_diff = evaluate_model(X_test, y_test, model, device)
            # get worst samples
            worst_samples = get_worst_samples(df_test, test_diff, config)

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

        # end of process for k, freeing memory
        # del model

        if keep_models:
            model_list[k] = copy.deepcopy(model)
            scaler_list[k] = copy.deepcopy(dataset_train.scaler)

    # Process is complete.

    return training_results, model_list, scaler_list
