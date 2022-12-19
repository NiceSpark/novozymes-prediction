import random
import torch
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from cuml import PCA

from .file_utils import open_json


class Hybrid_Dataset(Dataset):
    """
    Prepare the Novozymes dataset for regression
    we can give it a reverse_mutation_probability:
    => probability of giving the reverse mutation, not the direct one
    ie. mutated -> wild instead of wild-> mutated
    """

    def __init__(self, X: pd.DataFrame, y_ddG: np.ndarray, y_dTm: np.ndarray,
                 features_infos: dict, pca_n_components: int, reverse_probability: float, use_double=False):

        self.used_torch_type = torch.double if use_double else torch.float
        self.reverse_probability = reverse_probability
        self.reverse_coef = features_infos["reverse_coef"]
        self.direct_features = features_infos["direct_features"]
        self.indirect_features = features_infos["indirect_features"]

        # scaling and PCA
        self.X_scaler = StandardScaler()
        self.ddG_scaler = StandardScaler()
        self.dTm_scaler = StandardScaler()
        self.pca_direct = PCA(n_components=pca_n_components)
        self.pca_indirect = PCA(n_components=pca_n_components)
        # voxel features:
        self.X_voxel_features = X.pop("direct_voxel_features").to_numpy()
        # Apply scaling, X now contains all columns except voxel features
        self.X_features = self.X_scaler.fit_transform(X)
        # Apply PCA on regression features
        # direct mutation features
        self.X_direct_features = self.pca_direct.fit_transform(
            self.X_features[:, self.direct_features])
        # indirect mutation features, we need to reverse the delta features
        indirect_X = self.X_features[:,
                                     self.indirect_features]*self.reverse_coef
        self.X_indirect_features = self.pca_indirect.fit_transform(indirect_X)
        # y stays y
        self.y_ddG = self.ddG_scaler.fit_transform(y_ddG)
        self.y_dTm = self.dTm_scaler.fit_transform(y_dTm)

    def __len__(self):
        return len(self.X_features)

    def __getitem__(self, i):

        if random.random() > self.reverse_probability:
            # direct mutation
            if (type(self.X_voxel_features[i]) == np.float64
                    and self.X_voxel_features[i] == 0):
                x_voxel_features = torch.Tensor()
            else:
                x_voxel_features = torch.stack([torch.as_tensor(x, dtype=self.used_torch_type)
                                                for x in self.X_voxel_features[i]])
            x_features = torch.as_tensor(
                self.X_direct_features[i], dtype=self.used_torch_type)
            y_ddG = torch.as_tensor(self.y_ddG[i], dtype=self.used_torch_type)
            y_dTm = torch.as_tensor(self.y_dTm[i], dtype=self.used_torch_type)
        else:
            # we also need to reverse the voxel feature
            # TODO: check that
            x_voxel_features = np.concatenate(
                [self.X_voxel_features[i][7:], self.X_voxel_features[i][:7]], axis=0)
            # we also reverse the target
            y_ddG = -1*self.y_ddG[i]
            y_dTm = -1*self.y_dTm[i]

            # to torch tensor
            x_features = torch.as_tensor(
                self.X_indirect_features[i], dtype=self.used_torch_type)
            x_voxel_features = torch.stack([torch.as_tensor(x, dtype=self.used_torch_type)
                                            for x in x_voxel_features])
            y_ddG = torch.as_tensor(y_ddG, dtype=self.used_torch_type)
            y_dTm = torch.as_tensor(y_dTm, dtype=self.used_torch_type)

        return x_voxel_features, x_features, y_ddG, y_dTm


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

    if config["ddG_as_input"]:
        features.append("ddG")
        reverse_coef.append(1)
        direct_features.append(index)
        indirect_features.append(index)
        index += 1
    if config["dTm_as_input"]:
        features.append("dTm")
        reverse_coef.append(1)
        direct_features.append(index)
        indirect_features.append(index)
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
        k = config["kfold"]
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


def prepare_train_dataset(df: pd.DataFrame, config: dict,
                          features: list, features_infos: dict):
    """
    prepare the dataset
    NB: We do not split the data into train/test here, see split_dataset function
    """
    used_type = np.float64 if config["use_double"] else np.float32

    # 1. create X,y:
    X_train = df[["direct_voxel_features"] + features]
    y_train_ddG = df[["ddG"]].values.astype(used_type)
    y_train_dTm = df[["dTm"]].values.astype(used_type)

    # 2. load the dataset
    dataset = Hybrid_Dataset(X_train, y_train_ddG, y_train_dTm, features_infos,
                             config["pca_n_components"],
                             reverse_probability=config["reverse_probability"],
                             use_double=config["use_double"])

    return dataset


def prepare_eval_data(df: pd.DataFrame, config: dict, features: list,
                      features_infos: dict, train_scaler: StandardScaler,
                      pca_direct: PCA, submission=False):
    """
    prepare the dataset for testing only
    """
    used_type = np.float64 if config["use_double"] else np.float32
    used_torch_type = torch.double if config["use_double"] else torch.float

    # create X_voxel_features, X_features, y:
    X_voxel_features = df["direct_voxel_features"].to_numpy()
    X_voxel_features = torch.stack([torch.as_tensor(x, dtype=used_torch_type)
                                    for x in X_voxel_features])

    X_features = df[features]
    X_features = train_scaler.transform(X_features)
    X_features = X_features[:, features_infos["direct_features"]]
    X_features = pca_direct.transform(X_features)
    X_features = torch.as_tensor(X_features, dtype=used_torch_type)

    if submission:
        y_ddG = torch.as_tensor(
            [np.nan]*len(X_features), dtype=used_torch_type)
        y_dTm = torch.as_tensor(
            [np.nan]*len(X_features), dtype=used_torch_type)
    else:
        y_ddG = df[["ddG"]].values.astype(used_type)
        y_dTm = df[["dTm"]].values.astype(used_type)
        y_ddG = torch.as_tensor(y_ddG, dtype=used_torch_type)
        y_dTm = torch.as_tensor(y_dTm, dtype=used_torch_type)

    return X_voxel_features, X_features, y_ddG, y_dTm


def evaluate_model(X_voxel_features: torch.Tensor, X_features: torch.Tensor,
                   y_ddG: torch.Tensor, y_dTm: torch.Tensor,
                   dataset_train: Hybrid_Dataset, model, device):
    """
    evaluate the model
    X and y must be torch tensors
    """
    len_ddG = 0
    len_dTm = 0
    scaled_mse_ddG = None
    scaled_mse_dTm = None
    mse_ddG = 0
    mse_dTm = 0
    y_pred_ddG, y_pred_dTm = model(
        X_voxel_features.to(device), X_features.to(device))

    # we first compute the total mse, by adding the mse from ddG and dTm
    # we can do that here because ddG and dTm are scaled.
    # Moreover the scaling means that this mse cannot really be
    # linked to experimental data values

    if y_pred_ddG is not None:
        # to numpy
        y_pred_ddG = y_pred_ddG.cpu().detach().numpy()
        y_true_ddG = y_ddG.numpy()
        scaled_y_true_ddG = dataset_train.ddG_scaler.transform(y_true_ddG)
        # vstack everything
        y_pred_ddG = np.vstack(y_pred_ddG)
        y_true_ddG = np.vstack(y_true_ddG)
        scaled_y_true_ddG = np.vstack(scaled_y_true_ddG)
        # non nan ddG
        not_nan_ddG = ~np.isnan(y_true_ddG)
        len_ddG = not_nan_ddG.sum()
        scaled_mse_ddG = mean_squared_error(y_pred_ddG[not_nan_ddG],
                                            scaled_y_true_ddG[not_nan_ddG]
                                            ).astype(np.float64)
        # we now compute ddG MSE, this will not be scaled and make more sens
        y_pred_ddG = dataset_train.ddG_scaler.inverse_transform(y_pred_ddG)
        mse_ddG = mean_squared_error(y_pred_ddG[not_nan_ddG],
                                     y_true_ddG[not_nan_ddG]
                                     ).astype(np.float64)

    if y_pred_dTm is not None:
        # to numpy
        y_pred_dTm = y_pred_dTm.cpu().detach().numpy()
        y_true_dTm = y_dTm.numpy()
        scaled_y_true_dTm = dataset_train.dTm_scaler.transform(y_true_dTm)
        # vstack everything
        y_pred_dTm = np.vstack(y_pred_dTm)
        y_true_dTm = np.vstack(y_true_dTm)
        scaled_y_true_dTm = np.vstack(scaled_y_true_dTm)
        # non nan ddG
        not_nan_dTm = ~np.isnan(y_true_dTm)
        len_dTm = not_nan_dTm.sum()
        scaled_mse_dTm = mean_squared_error(y_pred_dTm[not_nan_dTm],
                                            scaled_y_true_dTm[not_nan_dTm]
                                            ).astype(np.float64)
        # we now compute dTm MSE, this will not be scaled and make more sens
        y_pred_dTm = dataset_train.dTm_scaler.inverse_transform(y_pred_dTm)
        mse_dTm = mean_squared_error(y_pred_dTm[not_nan_dTm],
                                     y_true_dTm[not_nan_dTm]
                                     ).astype(np.float64)

    if (scaled_mse_ddG is not None) and (scaled_mse_dTm is not None):
        scaled_mse = ((len_ddG*scaled_mse_ddG) +
                      (len_dTm*scaled_mse_dTm))/(len_ddG+len_dTm)
    elif scaled_mse_ddG is not None:
        scaled_mse = scaled_mse_ddG
    else:
        scaled_mse = scaled_mse_dTm
    # we inverse the transformation to get the real values of ddG and dTm

    return scaled_mse, mse_ddG, mse_dTm


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
    df = df[((df.ddG <= config.get("max_ddG_value", 0)) |
            (df.dTm <= config.get("max_dTm_value", 0)))]

    # keep only mutations with adequate targets
    if config["ddG_as_input"] or config["dTm_as_input"]:
        df = df[~(df.ddG.isna()) & ~(df.dTm.isna())]
    elif "ddG" not in config["targets"]:
        # if ddG not in targets we keep only mutations with dTm values
        df = df[~(df.dTm.isna())]
    elif "dTm" not in config["targets"]:
        # if dTm not in targets we keep only mutations with ddG values
        df = df[~(df.ddG.isna())]

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


def get_device(config):
    if torch.cuda.is_available() and config["use_cuda"]:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)
    return device
