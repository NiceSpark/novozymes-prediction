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
                optimizer, scheduler, loss_function,
                trainloader, device, dataset_train,
                X_test_voxel, X_test_features,
                y_test_ddG, y_test_dTm):
    """
    train the model
    we update some stastistic data over each epoch
    we also keep the best model (as a function of epoch)
    and we stop training short after a certain number of epochs
    """

    loss_over_time = []
    test_mse_over_time = []
    test_mse_ddG_over_time = []
    test_mse_dTm_over_time = []
    learning_rates = []
    best_test_mse = np.inf
    best_epoch = 0
    best_model = copy.deepcopy(model)

    # Run the training loop
    for epoch in range(config["num_epochs"]):

        # set model in train mode
        model.train()
        # Set current loss value
        current_loss = 0.0
        i = 0
        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):
            # losses are initialized as we have multiple targets
            loss_ddG = torch.zeros(1, requires_grad=True)
            loss_dTm = torch.zeros(1, requires_grad=True)
            # Get and prepare inputs
            voxel_inputs, features_inputs, ddG_targets, dTm_targets = data
            # Move to cuda device
            voxel_inputs = voxel_inputs.to(device)
            features_inputs = features_inputs.to(device)
            # separate targets, keep only non nan
            ddG_targets = ddG_targets.to(device)
            dTm_targets = dTm_targets.to(device)
            ddG_targets = ddG_targets.reshape((ddG_targets.shape[0], 1))
            dTm_targets = dTm_targets.reshape((dTm_targets.shape[0], 1))
            not_nan_ddG_targets = ~torch.isnan(ddG_targets)
            not_nan_dTm_targets = ~torch.isnan(dTm_targets)
            # Zero the gradients
            optimizer.zero_grad()
            # Perform forward pass
            ddG_outputs, dTm_outputs = model(voxel_inputs, features_inputs)
            # compute ddG and dTm loss
            if torch.any(not_nan_ddG_targets):
                loss_ddG = loss_function(ddG_outputs[not_nan_ddG_targets],
                                         ddG_targets[not_nan_ddG_targets])

            if torch.any(not_nan_dTm_targets):
                loss_dTm = loss_function(dTm_outputs[not_nan_dTm_targets],
                                         dTm_targets[not_nan_dTm_targets])
            # Compute global loss
            len_ddG, len_dTm = not_nan_ddG_targets.sum(), not_nan_dTm_targets.sum()
            loss = ((len_ddG*loss_ddG)+(len_dTm*loss_dTm))/(len_ddG+len_dTm)
            # Perform backward pass
            loss.backward()
            # Perform optimization
            optimizer.step()

            # compute statistics
            current_loss += loss.item()

        learning_rates.append(optimizer.param_groups[0]["lr"])
        loss_over_time.append(current_loss/(i+1))

        scheduler.step()

        model.eval()
        with torch.set_grad_enabled(False):
            test_mse, test_mse_ddG, test_mse_dTm = evaluate_model(X_test_voxel, X_test_features,
                                                                  y_test_ddG, y_test_dTm,
                                                                  dataset_train, model, device)
            test_mse_over_time.append(test_mse)
            test_mse_ddG_over_time.append(test_mse_ddG)
            test_mse_dTm_over_time.append(test_mse_dTm)

        if test_mse < best_test_mse:
            best_test_mse = test_mse
            best_epoch = epoch
            best_model = copy.deepcopy(model)
        if epoch - best_epoch > config["stop_train_epochs"]:
            print(f"Early stopping at epoch: {epoch}")
            break

    results = {
        "loss_over_time": loss_over_time,
        "test_mse_over_time": test_mse_over_time,
        "test_mse_ddG_over_time": test_mse_ddG_over_time,
        "test_mse_dTm_over_time": test_mse_dTm_over_time,
        "learning_rate_over_time": learning_rates,
        "best_epoch": best_epoch,
        "best_test_mse": best_test_mse,
    }

    return best_model, results


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


def build_scheduler(optimizer, config):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config.get("scheduler_milestones", [20, 30, 40]),
        gamma=config.get("scheduler_gamma", 0.5)
    )
    return scheduler


def k_fold_training(df, config, features, features_infos,
                    device, wandb_active=False, wandb_config={}, keep_models=False):
    training_results = []
    model_list = [None]*config["kfold"]
    scaler_list = [None]*config["kfold"]

    if wandb_active:
        wandb.init(config=wandb_config)
        # update config based on wandb.config
        config.update(wandb.config)

    used_torch_type = torch.double if config["use_double"] else torch.float

    for k in tqdm.tqdm(range(config["kfold"])):
        train = list(range(config["kfold"]))
        test = [train.pop(k)]
        df_train = df[df["kfold"].isin(train)]
        df_test = df[df["kfold"].isin(test)]

        # we load the data for training
        dataset_train = prepare_train_dataset(
            df_train, config, features, features_infos)
        trainloader = DataLoader(dataset_train,
                                 batch_size=config["batch_size"],
                                 shuffle=config["shuffle_dataloader"],
                                 num_workers=config["num_workers"])

        # we load the data for evaluation
        X_test_voxel, X_test_features, y_test_ddG, y_test_dTm = prepare_eval_data(
            df_test, config, features, features_infos, dataset_train.X_scaler)

        # Initialize a new Novozymes Model
        model = HybridNN(
            len(features_infos["direct_features"]), config)
        model.to(used_torch_type)
        model.to(device)

        # Define the loss function, optimizer and scheduler
        loss_function = nn.MSELoss()
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)
        # Train model:
        t0 = time.time()
        model, results_by_epochs = train_model(model, config, optimizer, scheduler,
                                               loss_function, trainloader, device,
                                               dataset_train,  # we need the scaler
                                               X_test_voxel, X_test_features,
                                               y_test_ddG, y_test_dTm)

        t1 = time.time()-t0

        # Evaluate this model:
        model.eval()
        with torch.set_grad_enabled(False):
            # print(f"MSE obtained for k-fold {k}: {mse}")
            results = results_by_epochs
            results.update({
                "time": t1
            })
            training_results.append(results)

        if keep_models:
            torch.save(model, f"tmp/model_{k}.pth")
            dump(dataset_train.X_scaler, open(f"tmp/X_scaler_{k}.pkl", "wb"))
            dump(dataset_train.ddG_scaler, open(
                f"tmp/ddG_scaler_{k}.pkl", "wb"))
            dump(dataset_train.dTm_scaler, open(
                f"tmp/dTm_scaler_{k}.pkl", "wb"))

        # end of process for k, freeing memory
        del model
        del df_train, df_test
        del dataset_train, trainloader
        del X_test_voxel, X_test_features, y_test_ddG, y_test_dTm
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
