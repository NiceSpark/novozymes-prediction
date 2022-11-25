import pandas as pd
from .file_utils import write_json


def add_demask_predictions_by_uniprot(df: pd.DataFrame, uniprot_id: str, errors: dict):
    prediction_path = f"./data/main_dataset_creation/DeMaSk_outputs/predictions/{uniprot_id}.txt"
    demask_df = pd.read_csv(prediction_path, sep='\t')

    def add_infos(row, uniprot_id, errors):
        if row["uniprot"] == uniprot_id:
            mutated_aa = str(row["mutated_aa"])
            if mutated_aa == '-':
                # this is a deletion mutation, no demask score available
                return row

            # demask index residue starting at 1
            pos = int(row["mutation_position"])+1
            wild_aa = str(row["wild_aa"])
            prediction = demask_df.loc[demask_df["pos"].eq(
                pos) & demask_df["WT"].eq(wild_aa) & demask_df["var"].eq(mutated_aa)]

            if len(prediction.index) != 1:
                # print("error: prediction contains more than one element")
                if row["uniprot"] not in errors.keys():
                    errors[row["uniprot"]] = []
                errors[row["uniprot"]].append({
                    "wild_aa": wild_aa,
                    "mutation_position": row["mutation_position"],
                    "pos": pos,
                    "mutated_aa": mutated_aa})
                return row

            row["direct_demask_score"] = prediction["score"].iloc[0]
            row["direct_demask_entropy"] = prediction["entropy"].iloc[0]
            row["direct_demask_log2f_var"] = prediction["log2f_var"].iloc[0]
            row["direct_demask_matrix"] = prediction["matrix"].iloc[0]
        return row

    df = df.apply(lambda row: add_infos(row, uniprot_id, errors), axis=1)
    return df, errors


def add_demask_predictions_by_mutation(df: pd.DataFrame, errors: dict):

    def add_infos(row, errors):
        mutation_position = int(row["mutation_position"])
        mutated_aa = str(row["mutated_aa"])
        wild_aa = str(row["wild_aa"])

        if mutated_aa == '-':
            # this is a deletion mutation, no demask score available
            return row

        try:
            name = row['uniprot'] + \
                f"_{row['wild_aa']}{mutation_position}{mutated_aa}"
            prediction_path = f"./data/main_dataset_creation/DeMaSk_outputs/predictions/{name}.txt"
            demask_df = pd.read_csv(prediction_path, sep='\t')

            # demask index residue starting at 1
            pos = mutation_position+1
            prediction = demask_df.loc[demask_df["pos"].eq(
                pos) & demask_df["WT"].eq(mutated_aa) & demask_df["var"].eq(wild_aa)]

            if len(prediction.index) != 1:
                # print("error: prediction contains more than one element")
                errors.update({row["uniprot"]: {"uniprot": row["uniprot"],
                                                "wild_aa": wild_aa,
                                                "mutation_position": row["mutation_position"],
                                                "pos": pos,
                                                "mutated_aa": mutated_aa}})
                return row

            row["indirect_demask_score"] = prediction["score"].iloc[0]
            row["indirect_demask_entropy"] = prediction["entropy"].iloc[0]
            row["indirect_demask_log2f_var"] = prediction["log2f_var"].iloc[0]
            row["indirect_demask_matrix"] = prediction["matrix"].iloc[0]
        except Exception as e:
            errors.update({row["uniprot"]: {"uniprot": row["uniprot"],
                                            "wild_aa": wild_aa,
                                            "mutation_position": row["mutation_position"],
                                            "mutated_aa": mutated_aa}})
            print(
                f"Exception raised for {row['uniprot']} {mutation_position} {mutated_aa}: {e}")
        return row

    df = df.apply(lambda row: add_infos(row, errors), axis=1)
    return df, errors


def add_demask_predictions(df: pd.DataFrame, multiprocessing=False):
    # we add the columns to the df
    if not multiprocessing:
        new_columns = [
            "direct_demask_entropy",
            "direct_demask_log2f_var",
            "direct_demask_matrix",
            "direct_demask_score",
            "indirect_demask_entropy",
            "indirect_demask_log2f_var",
            "indirect_demask_matrix",
            "indirect_demask_score"
        ]
        for column in new_columns:
            df[column] = 0.0

    errors = {}

    for uniprot_id in df.uniprot.unique():
        df, errors = add_demask_predictions_by_uniprot(df, uniprot_id, errors)

    df, errors = add_demask_predictions_by_mutation(df, errors)

    return df
