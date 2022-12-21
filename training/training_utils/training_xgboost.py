"""
This file contains the training function for XGBoost models.
"""


import numpy as np
import xgboost as xg
import wandb

from tqdm import tqdm
from pickle import dump
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from .model_utils import prepare_xgboost_data


def k_fold_training_xgboost(df, config: dict, features: list, features_infos: dict,
                            wandb_active=False, wandb_config={},
                            keep_models=False):
    # initialize the training results
    training_results = []

    if wandb_active:
        # initialize wandb
        wandb.init(config=wandb_config)
        # update config based on wandb.config
        config.update(wandb.config)

    for k in tqdm(range(config["kfold"])):
        # split the data into train and test
        train = list(range(config["kfold"]))
        test = [train.pop(k)]
        df_train = df[df["kfold"].isin(train)]
        df_test = df[df["kfold"].isin(test)]
        # we create a scaler for both training and testing
        X_scaler = StandardScaler()

        # we load the data
        X_train, y_train_ddG, y_train_dTm = prepare_xgboost_data(
            df_train, features, features_infos, X_scaler, fit_scaler=True)
        X_test, y_test_ddG, y_test_dTm = prepare_xgboost_data(
            df_test, features, features_infos, X_scaler, fit_scaler=False)
        # Initialize a new Novozymes Model
        model = xg.XGBRegressor(objective='reg:squarederror',
                                n_estimators=config["n_estimators"],
                                max_depth=config["max_depth"],
                                learning_rate=config["learning_rate"],
                                seed=42
                                )
        # Fitting the model
        # TODO: multi outputs model, for now we do not use dTm
        model.fit(X_train, y_train_ddG)
        # Evaluate this model:
        y_train_predicted_ddG = model.predict(X_train)
        y_test_predicted_ddG = model.predict(X_test)
        train_mse = mean_squared_error(y_train_ddG, y_train_predicted_ddG)
        test_mse = mean_squared_error(y_test_ddG, y_test_predicted_ddG)

        results = {
            "train_mse": train_mse,
            "test_mse": test_mse
        }
        training_results.append(results)

        if keep_models:
            # Save the model and the scaler in tmp/
            dump(X_scaler, open(f"tmp/X_scaler_{k}.pkl", "wb"))
            model.save_model(f"tmp/model_{k}.json")

    # Process is complete.
    if wandb_active:
        # log the results in wandb
        avg_test_mse = np.mean(
            np.array([r["test_mse"] for r in training_results]), axis=0)
        avg_train_mse = np.mean(
            np.array([r["train_mse"] for r in training_results]), axis=0)

        wandb.log({
            "test_mse": avg_test_mse,
            "train_mse": avg_train_mse,
        })
        # finish the wandb run
        wandb.finish()

    return training_results
