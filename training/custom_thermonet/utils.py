import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ThermoNet_Dataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: np.ndarray, features: list):
        self.X = X[features].to_numpy()
        self.y = y

    def __getitem__(self, index):
        y_data = self.y[index]
        x_data = torch.stack([torch.as_tensor(x, dtype=torch.float)
                              for x in self.X[index, 0]])
        y_data = torch.as_tensor(y_data, dtype=torch.float)

        return x_data, y_data

    def __len__(self):
        return len(self.X)


def thermonet_train_data(df: pd.DataFrame, config: dict, features: list):
    """
    prepare the dataset
    NB: We do not split the data into train/test here, see split_dataset function
    """
    # 1. get the target
    target = config["target"]

    # 2. create X,y:
    X_train = df[features]
    y_train = df[[target]].values.astype(float)

    # 3. load the dataset
    dataset = ThermoNet_Dataset(X_train, y_train, features)

    return dataset


def thermonet_eval_data(df: pd.DataFrame, config: dict, features: list):
    """
    prepare the dataset for testing only
    """
    # 1. get the target
    target = config["target"]

    # 2. create X,y:
    X_eval = df[features].to_numpy()

    y_eval = df[[target]].values.astype(float)

    X_eval = torch.stack([torch.as_tensor(x, dtype=torch.float)
                         for x in X_eval[:, 0]])
    y_eval = torch.as_tensor(y_eval, dtype=torch.float)

    return X_eval, y_eval
