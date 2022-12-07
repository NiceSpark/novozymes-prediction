import random
import torch
import shutil
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from pickle import load
from glob import glob

from .file_utils import open_json


class Hybrid_Dataset(Dataset):
    """
    Prepare the Novozymes dataset for regression
    we can give it a reverse_mutation_probability:
    => probability of giving the reverse mutation, not the direct one
    ie. mutated -> wild instead of wild-> mutated
    """

    def __init__(self, X: pd.DataFrame, y: np.ndarray,
                 features_infos: dict, reverse_probability: float, use_double=False):

        self.scaler = StandardScaler()
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            # voxel features:
            self.X_voxel_features = X.pop("direct_voxel_features").to_numpy()
            # Apply scaling, X now contains all columns except voxel features
            self.X_features = self.scaler.fit_transform(X)
            # y stays y
            self.y = y

            self.used_torch_type = torch.double if use_double else torch.float
            self.reverse_probability = reverse_probability
            self.reverse_coef = features_infos["reverse_coef"]
            self.direct_features = features_infos["direct_features"]
            self.indirect_features = features_infos["indirect_features"]

    def __len__(self):
        return len(self.X_features)

    def __getitem__(self, i):

        if random.random() > self.reverse_probability:
            # direct mutation
            if (type(self.X_voxel_features[i]) == type(0.0)
                    and self.X_voxel_features[i] == 0):
                x_voxel_features = torch.Tensor()
            else:
                x_voxel_features = torch.stack([torch.as_tensor(x, dtype=self.used_torch_type)
                                                for x in self.X_voxel_features[i]])
            x_features = torch.as_tensor(
                self.X_features[i][self.direct_features], dtype=self.used_torch_type)
            y = torch.as_tensor(self.y[i], dtype=self.used_torch_type)
        else:
            # indirect mutation, we need to reverse the delta features
            all_features = self.X_features[i]
            reverse = np.multiply(all_features, np.array(self.reverse_coef))
            x_features = reverse[self.indirect_features]
            # we also need to reverse the voxel feature
            # TODO: check that
            x_voxel_features = np.concatenate(
                [self.X_voxel_features[i][7:], self.X_voxel_features[i][:7]], axis=0)
            # we also reverse the target
            y = -1*self.y[i]

            # to torch tensor
            x_features = torch.as_tensor(
                x_features, dtype=self.used_torch_type)
            x_voxel_features = torch.stack([torch.as_tensor(x, dtype=self.used_torch_type)
                                            for x in x_voxel_features])
            y = torch.as_tensor(y, dtype=self.used_torch_type)

        return x_voxel_features, x_features, y


def compute_feature_list(config: dict, features_dict: dict):
    # compute feature list
    features_config = config["features_config"]
    reverse_coef = []
    direct_features = []
    indirect_features = []

    features = []
    index = 0

    for key, features_sublist in features_dict.items():
        if features_config[key].get("use"):
            for feature in features_sublist:
                category = features_config[key].get("category")
                if category == "global":
                    reverse_coef.append(1)
                    direct_features.append(index)
                    indirect_features.append(index)
                elif category == "direct":
                    reverse_coef.append(1)
                    direct_features.append(index)
                elif category == "indirect":
                    reverse_coef.append(1)
                    indirect_features.append(index)
                if category == "delta":
                    reverse_coef.append(-1)
                    direct_features.append(index)
                    indirect_features.append(index)
                features.append(feature)
                index += 1

    features_infos = {
        "reverse_coef": reverse_coef,
        "direct_features": direct_features,
        "indirect_features": indirect_features,
    }

    return features, features_infos


def split_dataset(df: pd.DataFrame, config):
    """adds the kfold group to each row, based on the fixed_ksplit from the config"""
    if os.path.exists(config["ksplit_path"]):
        fixed_ksplit = open_json(config["ksplit_path"])
        df["kfold"] = df["alphafold_path"].apply(
            lambda x: fixed_ksplit[x])
    else:
        print("no valid ksplit path given, doing ksplit without groups")
        df.reset_index(drop=True, inplace=True)
        k = 5
        kfold = KFold(k, shuffle=True)
        split = list(kfold.split(range(len(df))))

        def get_fold_id(i, split):
            for k in range(len(split)):
                if i in split[k][1]:
                    return k
            return np.nan

        df["kfold"] = df.apply(
            lambda row: get_fold_id(row.name, split), axis=1)

    return df


def prepare_train_data(df: pd.DataFrame, config: dict,
                       features: list, features_infos: dict):
    """
    prepare the dataset
    NB: We do not split the data into train/test here, see split_dataset function
    """
    used_type = np.float64 if config["use_double"] else np.float32

    # 1. get the target
    target = config["target"]

    # 2. create X,y:
    X_train = df[["direct_voxel_features"] + features]
    y_train = df[[target]].values.astype(used_type)

    # 3. load the dataset
    dataset = Hybrid_Dataset(X_train, y_train, features_infos,
                             reverse_probability=config["reverse_probability"],
                             use_double=config["use_double"])

    return dataset


def prepare_eval_data(df: pd.DataFrame, config: dict, features: list,
                      features_infos: dict, train_scaler: StandardScaler,
                      submission=False):
    """
    prepare the dataset for testing only
    """
    used_type = np.float64 if config["use_double"] else np.float32
    used_torch_type = torch.double if config["use_double"] else torch.float

    # 1. get the target
    target = config["target"]

    # 2. create X_voxel_features, X_features, y:
    X_voxel_features = df["direct_voxel_features"].to_numpy()
    X_voxel_features = torch.stack([torch.as_tensor(x, dtype=used_torch_type)
                                    for x in X_voxel_features])

    X_features = df[features]
    X_features = train_scaler.transform(X_features)
    X_features = torch.from_numpy(X_features)
    X_features = X_features[:, features_infos["direct_features"]]
    X_features = torch.as_tensor(X_features, dtype=used_torch_type)

    if submission:
        y = torch.as_tensor([np.nan]*len(X_features), dtype=used_torch_type)
    else:
        y = df[[target]].values.astype(used_type)
        y = torch.as_tensor(y, dtype=used_torch_type)

    return X_voxel_features, X_features, y


def evaluate_model(X_voxel_features: torch.Tensor, X_features: torch.Tensor,
                   y: torch.Tensor, model, device):
    """
    evaluate the model
    X and y must be torch tensors
    """
    y_pred = model(X_voxel_features.to(device), X_features.to(device))
    y_pred = y_pred.cpu().detach().numpy()
    y_pred = np.vstack(y_pred)

    y_true = y.numpy()
    y_true = np.vstack(y_true)

    # compute MSE
    mse = mean_squared_error(y_true, y_pred)
    mse = mse.astype(np.float64)
    # compute worst results
    diff = np.abs(np.diff(np.array([y_true, y_pred]), axis=0))[0]

    return mse, diff


def get_worst_samples(df_test, test_diff, config):
    uniprot_df_test = df_test[["uniprot", "wild_aa",
                              "mutation_position", "mutated_aa"]].copy()
    uniprot_df_test["diff"] = test_diff
    worst_samples = uniprot_df_test.nlargest(
        config["num_worst_samples"], ["diff"]).to_dict("records")
    return worst_samples


def load_dataset(config, features, rm_nan=False):
    df = pd.read_csv(f"{config['dataset_dir']}/{config['dataset_name']}.csv")

    # remove bad uniprot
    df = df[~(df["uniprot"].isin(config["bad_uniprot"]))]

    if rm_nan:
        for feature in features:
            df = df[~(df[feature].isna())]

    # keep only mutations with adequate ddG value
    # usually mutations are destabilizing, so we guess in the submission dataset they are
        df = df[(df.ddG <= config.get("max_ddG_value", 0))]

    # apply max protein length
    df = df[df.length.le(config["max_protein_length"])]

    if config["model_type"] in ["hybrid", "cnn_only"]:
        # load voxel features
        if config["use_pdb_chain_voxel"]:
            print("using pdb_chain_voxel")
            df = df[~(df["pdb_chain_voxel_path"].isna())]
            df["direct_voxel_features"] = df["pdb_chain_voxel_path"].apply(
                np.load)
        else:
            df["direct_voxel_features"] = df["direct_voxel_path"].apply(
                np.load)
    else:
        df["direct_voxel_features"] = 0.0

    print(f"loaded {len(df)} data")
    return df


def move_models_and_scalers(dir_path: str):
    # we save models and scalers during training in tmp/,
    # we now need to move them to the appropriated folder
    for model_tmp_path in glob("tmp/model*.pth"):
        shutil.move(model_tmp_path, f"{dir_path}/")
    for scaler_tmp_path in glob("tmp/scaler*.pkl"):
        shutil.move(scaler_tmp_path, f"{dir_path}/")


def load_models_and_scalers(dir_path: str):
    model_list, scaler_list = [], []

    for k in range(len(glob(f"{dir_path}/model_*.pth"))):
        model_list.append(torch.load(f"{dir_path}/model_{k}.pth"))
    for k in range(len(glob(f"{dir_path}/scaler_*.pkl"))):
        scaler_list.append(load(open(f"{dir_path}/scaler_{k}.pkl", 'rb')))

    return model_list, scaler_list


def get_device(config):
    if torch.cuda.is_available() and config["use_cuda"]:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)
    return device
