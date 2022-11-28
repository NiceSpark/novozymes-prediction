import pandas as pd
import numpy as np
from glob import glob


def get_thermonet_df(name: str, base_dir="./compute_mutated_structures/"):
    """
    function that creates a df identification infos and the avg prediction values of thermonet
    (both direct and reversed)

    returns:
    a pandas.DataFrame which has the columns:
    ['wild_path', 'position', 'mutated_path', 'direct_thermonet', 'reversed_thermonet']
    """

    variant_path = base_dir+"gends_input/"+name+"_variants.txt"
    base_preds_path = base_dir+"thermonet_predictions/"+name

    # first we get the variant list in a df, in order to know to which mutations each value corresponds to
    variant_df = pd.read_csv(variant_path,
                             names=["wild_path", "position", "mutated_path"],
                             sep=' ')
    variant_df.position = variant_df.position.apply(lambda x: x-1).astype(int)

    # then we get the predictions from the thermonet outputs
    direct_preds = np.zeros((len(variant_df), 10))
    reversed_preds = np.zeros((len(variant_df), 10))
    try:
        for i in range(10):
            direct_preds[:, i] = np.loadtxt(
                base_preds_path+f"_direct_prediction_{i+1}.txt")
            reversed_preds[:, i] = np.loadtxt(
                base_preds_path+f"_reversed_prediction_{i+1}.txt")
    except Exception as e:
        print(f"Exception raised for {name}, {base_preds_path}: {e}")
        print(f"not adding infos for {name}")
        return pd.DataFrame()

    # thermonet outputs 10 predictions by mutation, we need to take the avg (=mean here)
    variant_df["direct_thermonet"] = np.mean(direct_preds, axis=1)
    variant_df["reversed_thermonet"] = np.mean(reversed_preds, axis=1)

    # we return the variant df which has the columns:
    # ['wild_path', 'position', 'mutated_path', 'direct_thermonet', 'reversed_thermonet']

    return variant_df


def update_main_df(row, main_df: pd.DataFrame):
    # we get mutated_path as a unique protein+mutation identifier
    # but multiple record could have the same mutation on the same protein
    # for example same mutation at different pH

    mutated_path = row["mutated_path"]
    main_df.loc[
        (main_df.relaxed_mutated_3D_path.eq(mutated_path)),
        "direct_thermonet"
    ] = row["direct_thermonet"]
    main_df.loc[
        (main_df.relaxed_mutated_3D_path.eq(mutated_path)),
        "reversed_thermonet"
    ] = row["reversed_thermonet"]

    return row


def add_thermonet_predictions(main_df: pd.DataFrame, base_dir="./compute_mutated_structures/"):
    new_columns_df = pd.DataFrame(
        columns=["direct_thermonet", "reversed_thermonet"])
    main_df = pd.concat([main_df, new_columns_df], axis=1)

    all_variants = glob(
        f"{base_dir}gends_input/*_variants.txt")
    all_names = [v.split('/')[-1].split("_variants.txt")[0]
                 for v in all_variants]
    for name in all_names:
        df = get_thermonet_df(name, base_dir=base_dir)
        df.apply(lambda row: update_main_df(row, main_df), axis=1)

    return main_df
