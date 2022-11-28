import torch
import copy
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
import gc

from .utils import thermonet_train_data, thermonet_eval_data
from .model import ThermoNet2


def evaluate_model(X: torch.Tensor, y: torch.Tensor, model, device):
    """
    evaluate the model
    X and y must be torch tensors
    """
    y_pred = model(X.to(device))
    y_pred = y_pred.cpu().detach().numpy()
    y_pred = np.vstack(y_pred)

    y_true = y.numpy()
    y_true = np.vstack(y_true)

    # compute MSE
    mse = mean_squared_error(y_true, y_pred)

    return mse


def train_model(model: torch.nn.Module, config, optimizer, loss_function,
                trainloader, device,
                X_test, y_test):

    loss_over_time = []
    train_mse_over_time = []
    test_mse_over_time = []

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
                test_mse = evaluate_model(X_test, y_test, model, device)
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


def k_fold_training(df, global_config, features,
                    device, keep_models=False):
    training_results = []
    model_list = [None]*global_config["kfold"]

    config = global_config

    for k in tqdm(range(global_config["kfold"])):
        train = list(range(global_config["kfold"]))
        test = [train.pop(k)]
        df_train = df[df["kfold"].isin(train)]
        df_test = df[df["kfold"].isin(test)]

        # we load the data for training
        dataset_train = thermonet_train_data(df_train, config, features)
        trainloader = DataLoader(dataset_train,
                                 batch_size=config["batch_size"],
                                 shuffle=config["shuffle_dataloader"],
                                 num_workers=global_config["num_workers"])

        # we load the data for evaluation
        X_train, y_train = thermonet_eval_data(df_train, config, features)
        X_test, y_test = thermonet_eval_data(df_test, config, features)

        # Initialize a new Novozymes Model
        model = ThermoNet2(config)
        model.to(torch.float)
        model.to(device)

        # Define the loss function and optimizer
        loss_function = torch.nn.MSELoss()
        optimizer = build_optimizer(model, config)
        # Train model:
        model, loss_over_time, train_mse_over_time, test_mse_over_time = train_model(
            model, config, optimizer, loss_function,
            trainloader, device, X_test, y_test)

        # Evaluate this model:
        model.eval()
        with torch.set_grad_enabled(False):
            train_mse = evaluate_model(X_train, y_train, model, device)
            test_mse = evaluate_model(X_test, y_test, model, device)

            # print(f"MSE obtained for k-fold {k}: {mse}")
            results = {
                "loss_over_time": loss_over_time,
                "train_mse_over_time": train_mse_over_time,
                "test_mse_over_time": test_mse_over_time,
                "train_mse": train_mse,
                "test_mse": test_mse,
            }
            training_results.append(results)

        if keep_models:
            model_list[k] = copy.deepcopy(model)

        # end of process for k, freeing memory
        del model
        del dataset_train, trainloader
        del X_train, y_train
        del X_test, y_test
        gc.collect()
        torch.cuda.empty_cache()

    return training_results, model_list
