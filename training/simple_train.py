from training_utils.file_utils import open_json, log_kfold_training
from training_utils.models import HybridNN
from training_utils.model_utils import *
from training_utils.training import k_fold_training

KEEP_MODELS = True

config = open_json("hybrid_nn_config.json")
features_dict = open_json(
    f"{config['dataset_dir']}/{config['features_name']}.json")
features, features_infos = compute_feature_list(config, features_dict)
device = get_device(config)
log_name = config["model_type"]

print(len(features))
print(len(features_infos["direct_features"]))
print(config["batch_size"])

df = load_dataset(config, features, rm_nan=True)

all_training_results = {"simple_train": [],
                        "total_training_time": 0}

# add protein_index to the dataset and get ksplit:
df = split_dataset(df, config)
# training
training_results = k_fold_training(
    df, config, features, features_infos, device, keep_models=KEEP_MODELS)

# add training results to all the other ones
all_training_results["simple_train"] = training_results

# save results to output
model = HybridNN(
    len(features_infos["direct_features"]), config)
model_structure = str(model).replace(
    '(', '').replace(')', '').split('\n')
dir_path = log_kfold_training(
    log_name, all_training_results, config, features, model_structure)

if KEEP_MODELS:
    move_models_and_scalers(dir_path)

print(f"logged training in {dir_path}")
