import os
import numpy as np
from glob import glob

from .file_utils import open_json


def add_scores(row, scores, prefix):
    for key, value in scores.items():
        if key == "decoy":
            continue
        row[f"{prefix}_{key}"] = value
    return row


def add_delta_scores(row, wild_scores, mutated_scores, prefix):
    for key, mut_value in mutated_scores.items():
        if key == "decoy":
            continue
        row[f"{prefix}_{key}"] = mut_value-wild_scores[key]
    return row


def add_columns(df):
    new_columns = ["decoy", "dslf_fa13", "fa_atr", "fa_dun", "fa_elec", "fa_intra_rep", "fa_intra_sol_xover4", "fa_rep", "fa_sol",
                   "hbond_bb_sc", "hbond_lr_bb", "hbond_sc", "hbond_sr_bb", "linear_chainbreak", "lk_ball_wtd", "omega",
                   "overlap_chainbreak", "p_aa_pp", "pro_close", "rama_prepro", "ref", "total_score", "yhh_planarity"]
    prefixes = ["alphafold", "wild_relaxed", "mutated_relaxed", "mutation"]

    for prefix in prefixes:
        for col in new_columns:
            df[f"{prefix}_{col}"] = np.nan
    return df


def add_rosetta_scores_to_row(row, all_relaxed_scores, all_alphafold_scores):
    name, _ = os.path.splitext(row["alphafold_path"].split("/")[-1])
    alphafold_sc_path = f"./data/main_dataset_creation/3D_structures/alphafold/{name}.sc"
    wild_sc_path = f"./compute_mutated_structures/relaxed_pdb/{name}_relaxed/{name}_relaxed.sc"

    w_aa, m_aa = row["wild_aa"], row["mutated_aa"]
    pos = int(row["mutation_position"])+1
    mutated_sc_path = (f"./compute_mutated_structures/relaxed_pdb/{name}_relaxed/" +
                       f"{name}_relaxed_{w_aa}{pos}{m_aa}_relaxed.sc")
    if not os.path.exists(mutated_sc_path):
        mutated_sc_path = row["relaxed_mutated_3D_path"].replace(".pdb", ".sc")

    if alphafold_sc_path in all_alphafold_scores:
        alphafold_scores = open_json(alphafold_sc_path)
        row = add_scores(row, alphafold_scores, "alphafold")

    if (wild_sc_path in all_relaxed_scores) and (mutated_sc_path in all_relaxed_scores):
        wild_scores = open_json(wild_sc_path)
        mutated_scores = open_json(mutated_sc_path)
        row = add_scores(row, wild_scores, "wild_relaxed")
        row = add_scores(row, mutated_scores, "mutated_relaxed")
        row = add_delta_scores(row, wild_scores, mutated_scores, "mutation")
    elif (wild_sc_path in all_relaxed_scores):
        wild_scores = open_json(wild_sc_path)
        row = add_scores(row, wild_scores, "wild_relaxed")

    return row


def add_rosetta_scores(df, multiprocessing=False):
    if not multiprocessing:
        df = add_columns(df)

    all_relaxed_scores = glob(
        "./compute_mutated_structures/relaxed_pdb/**/*.sc")
    all_alphafold_scores = glob(
        "./data/main_dataset_creation/3D_structures/alphafold/*.sc")

    df = df.apply(lambda row: add_rosetta_scores_to_row(
        row, all_relaxed_scores, all_alphafold_scores),
        axis=1)
    return df
