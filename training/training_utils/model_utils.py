import numpy as np
import pandas as pd
import random
import torch
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from pickle import dump, load
from glob import glob

from .file_utils import open_json


class Row_Selector():
    """
    We want to be able to choose btw:
    - each protein is seen the same amount of time
    [OR]
    - each mutation, independantly from the protein, is seen the same amount of time
    """

    def __init__(self, df: pd.DataFrame, select_by_protein=False):
        self.select_by_protein = select_by_protein
        if self.select_by_protein:
            self.protein_index = df.protein_index.unique()
            self.protein_index_to_row_index = {
                _p: df[df.protein_index.eq(_p)].index for _p in self.protein_index}

    def get_index(self, i):
        if self.select_by_protein:
            # each protein is seen the same amount of time
            k = random.choice(self.protein_index)  # choose a random protein
            # choose a random mutation of this protein
            return random.choice(self.protein_index_to_row_index[k])
        else:
            # each mutation, independantly from the protein, is seen the same amount of time
            return i


class Novozymes_Dataset(Dataset):
    """
    Prepare the Novozymes dataset for regression
    we can give it a reverse_mutation_probability:
    => probability of giving the reverse mutation, not the direct one
    ie. mutated -> wild instead of wild-> mutated
    """

    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, row_selector: Row_Selector,
                 features_infos: dict, reverse_probability: float, scale_data=True):

        self.scaler = StandardScaler()
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            # Apply scaling if necessary
            if scale_data:
                X = self.scaler.fit_transform(X)
            self.X = X
            self.y = y
            self.row_selector = row_selector

            self.reverse_probability = reverse_probability
            self.reverse_coef = features_infos["reverse_coef"]
            self.direct_features = features_infos["direct_features"]
            self.indirect_features = features_infos["indirect_features"]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        index = self.row_selector.get_index(i)

        if random.random() > self.reverse_probability:
            # direct mutation
            x_data = self.X[index][self.direct_features]
            y_data = self.y[index]
        else:
            # indirect mutation, we need to reverse the delta features
            all_features = self.X[index]
            reverse = np.multiply(all_features, np.array(self.reverse_coef))
            x_data = reverse[self.indirect_features]
            # we also reverse the target
            y_data = -1*self.y[index]

        x_data = torch.from_numpy(x_data)
        y_data = torch.from_numpy(y_data)

        return x_data, y_data


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
    fixed_ksplit = open_json(config["ksplit_path"])
    df["kfold"] = df["alphafold_path"].apply(
        lambda x: fixed_ksplit[x])

    return df


def PolynomialFeatures_labeled(input_df, power):
    '''Basically this is a cover for the sklearn preprocessing function.
    The problem with that function is if you give it a labeled dataframe, it ouputs an unlabeled dataframe with potentially
    a whole bunch of unlabeled columns.
    Inputs:
    input_df = Your labeled pandas dataframe (list of x's not raised to any power)
    power = what order polynomial you want variables up to. (use the same power as you want entered into pp.PolynomialFeatures(power) directly)
    if power == 1, simply return input_df unchanged

    Output: This function relies on the powers_ matrix which is one of the preprocessing function's outputs to create logical labels and
    outputs a labeled pandas dataframe
    '''

    if power == 1:
        return input_df

    poly = PolynomialFeatures(power)
    output_nparray = poly.fit_transform(input_df)
    powers_nparray = poly.powers_

    input_feature_names = list(input_df.columns)
    target_feature_names = ["Constant Term"]
    for feature_distillation in powers_nparray[1:]:
        intermediary_label = ""
        final_label = ""
        for i in range(len(input_feature_names)):
            if feature_distillation[i] == 0:
                continue
            else:
                variable = input_feature_names[i]
                power = feature_distillation[i]
                intermediary_label = "%s^%d" % (variable, power)
                if final_label == "":  # If the final label isn't yet specified
                    final_label = intermediary_label
                else:
                    final_label = final_label + " x " + intermediary_label
        target_feature_names.append(final_label)
    output_df = pd.DataFrame(output_nparray, columns=target_feature_names)
    output_df.drop(columns=["Constant Term"], inplace=True)
    return output_df


def prepare_train_data(df: pd.DataFrame, config: dict,
                       features: list, features_infos: dict):
    """
    prepare the dataset
    NB: We do not split the data into train/test here, see split_dataset function
    """
    # 1. get the target
    target = config["target"]

    # 2. create X,y:
    X_train = PolynomialFeatures_labeled(
        df[features], config["polynomial_features_power"])
    y_train = df[[target]].values.astype(float)

    if (X_train.isna().sum().sum() > 0 or np.isnan(y_train).sum() > 0):
        print(
            f"ERROR: there are {X_train.isna().sum().sum()} na occurences in the X_train df")
        print(
            f"ERROR: there are {np.isnan(y_train).sum()} na occurences in the y_train df")

    # 3. create the row_selector object
    row_selector = Row_Selector(df, config["select_by_protein"])
    # 4. load the dataset
    dataset = Novozymes_Dataset(X_train, y_train, row_selector,
                                features_infos, reverse_probability=config["reverse_probability"])

    return dataset


def prepare_eval_data(df: pd.DataFrame, config: dict, features: list, features_infos: dict, train_scaler: StandardScaler):
    """
    prepare the dataset for testing only
    """
    # 1. get the target
    target = config["target"]

    # 2. create X,y:
    X_eval = PolynomialFeatures_labeled(
        df[features], config["polynomial_features_power"])

    y_eval = df[[target]].values.astype(float)
    X_eval = train_scaler.transform(X_eval)

    if (np.isnan(X_eval).sum() > 0 or np.isnan(y_eval).sum() > 0):
        print(
            f"ERROR: there are {np.isnan(X_eval).sum()} na occurences in the X_eval df")
        print(
            f"ERROR: there are {np.isnan(y_eval).sum()} na occurences in the y_eval df")
    X_eval = torch.from_numpy(X_eval)
    X_eval = X_eval[:, features_infos["direct_features"]]
    y_eval = torch.from_numpy(y_eval)

    return X_eval, y_eval


def prepare_xgboost(df, config, features, train_scaler=StandardScaler, fit_scaler=False):
    """
    prepare the dataset for testing only
    """
    # 1. get the target
    target = config["target"]

    # 2. create X,y:
    X = PolynomialFeatures_labeled(
        df[features], config["polynomial_features_power"])

    y = df[[target]].values.astype(float)
    if fit_scaler:
        train_scaler.fit(X)
    X = train_scaler.transform(X)

    if (np.isnan(X).sum() > 0 or np.isnan(y).sum() > 0):
        print(
            f"ERROR: there are {np.isnan(X).sum()} na occurences in the X df")
        print(
            f"ERROR: there are {np.isnan(y).sum()} na occurences in the y df")

    return X, y


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


def predict(row, model, device):
    """make a class prediction for one row of data"""
    # convert row to data
    row = torch.tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat


def load_dataset(config, features):
    df = pd.read_csv(f"{config['dataset_dir']}/{config['dataset_name']}.csv")

    # remove bad uniprot
    df = df[~(df["uniprot"].isin(config["bad_uniprot"]))]

    # apply max protein length
    df = df[df.length.le(config["max_protein_length"])]

    print(f"training on {len(df)} data")
    return df


def save_models_and_scalers(dir_path: str, model_list: list, scaler_list: list):
    for i, model in enumerate(model_list):
        torch.save(model, f"{dir_path}/model_{i}.pth")
    for i, scaler in enumerate(scaler_list):
        dump(scaler, open(f"{dir_path}/scaler_{i}.pkl", "wb"))


def load_models_and_scalers(dir_path: str):
    model_list, scaler_list = [], []

    for model_path in glob(f"{dir_path}/model_*.pth"):
        model_list.append(torch.load(model_path))
    for scaler_path in glob(f"{dir_path}/scaler_*.pkl"):
        scaler_list.append(load(open(scaler_path, 'rb')))

    return model_list, scaler_list
