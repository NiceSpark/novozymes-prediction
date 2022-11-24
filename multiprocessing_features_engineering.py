# This script does the same as features_engineering NB but does it with multiprocessing to go quicker
# we do this by splitting the df by uniprot, and then running the different stuff on each df
#
# We do not compute ESM features here, as we cannot multiprocessed cuda instances, therefore the ESM shall be done first
# Notice that the input dataset is the "with_esm_features" one

import pandas as pd
import numpy as np
import tqdm
from multiprocessing import Pool

from utils.features_biopython import add_structure_infos, add_protein_analysis, DSSP_Data_Keys
from utils.demask_features import add_demask_predictions
from utils.file_utils import open_json
from utils.infos_translation import aa_char2int
from utils.rosetta_features import add_rosetta_scores

THREADS = 12
MAX_SUBDF_SIZE = 200

ADD_STRUCTURE_INFOS = True
ADD_DEMASK_PREDICTIONS = True
ADD_ROSETTA_SCORES = True
ADD_PROTEIN_ANALYSIS = True
ONLY_DDG = True
SAVE_NEW_CSV = True
CONVERT_MUTATION_TO_INT = True
CLEAN_DF = True
START_FRESH = True

SUBMISSION = True

NAME = "all_v3"
DATASET_DIR = f"./data/main_dataset_creation/outputs/{NAME}/"
if SUBMISSION:
    DATASET_INPUT_PATH = DATASET_DIR+"submission_with_esm_features.csv"
    DATASET_OUTPUT_PATH = DATASET_DIR+"submission_with_all_features.csv"
else:
    DATASET_INPUT_PATH = DATASET_DIR+"dataset_with_structure_features.csv"
    DATASET_OUTPUT_PATH = DATASET_DIR+"dataset_with_all_features.csv"


if START_FRESH:
    df = pd.read_csv(DATASET_INPUT_PATH)
else:
    df = pd.read_csv(DATASET_OUTPUT_PATH)

# df.head(2)

if CLEAN_DF:
    print(len(df))
    if not SUBMISSION:
        if ONLY_DDG:
            # we drop rows without ddG
            df = df[~(df.ddG.isna())]
        # we drop rows without essential values
        for k in ["wild_aa", "mutation_position", "mutated_aa", "sequence", "alphafold_path", "relaxed_wild_3D_path", "relaxed_mutated_3D_path"]:
            df = df[~(df[k].isna())]
        print(len(df))
        # print(df.isna().sum().to_dict())


# ADD all columns that will be computed
new_columns = []
if ADD_DEMASK_PREDICTIONS:
    new_columns += [
        "direct_demask_entropy",
        "direct_demask_log2f_var",
        "direct_demask_matrix",
        "direct_demask_score",
        "indirect_demask_entropy",
        "indirect_demask_log2f_var",
        "indirect_demask_matrix",
        "indirect_demask_score"
    ]

if ADD_STRUCTURE_INFOS:
    for prefix in ["alphafold", "wild_relaxed", "mutated_relaxed", "mutation"]:
        new_columns += [f"{prefix}_{k}" for k in DSSP_Data_Keys[2:]]
        new_columns += [f"{prefix}_{k}" for k in ["sasa", "residue_depth",
                                                  "c_alpha_depth", "bfactor"]]

if ADD_ROSETTA_SCORES:
    columns = ["dslf_fa13", "fa_atr", "fa_dun", "fa_elec", "fa_intra_rep", "fa_intra_sol_xover4", "fa_rep", "fa_sol",
               "hbond_bb_sc", "hbond_lr_bb", "hbond_sc", "hbond_sr_bb", "linear_chainbreak", "lk_ball_wtd", "omega",
               "overlap_chainbreak", "p_aa_pp", "pro_close", "rama_prepro", "ref", "total_score", "yhh_planarity"]
    prefixes = ["alphafold", "wild_relaxed", "mutated_relaxed", "mutation"]

    for prefix in prefixes:
        for col in columns:
            new_columns.append(f"{prefix}_{col}")

if ADD_PROTEIN_ANALYSIS:
    new_columns += [
        "wild_wildcharge_at_pH",
        "wild_flexibility",
        "wild_gravy",
        "wild_molar_extinction_2",
        "wild_molar_extinction_1",
        "wild_sheet_fraction",
        "wild_turn_fraction",
        "wild_helix_fraction",
        "wild_isoelectric_point",
        "wild_aromaticity",
        "wild_instability_index",
        "wild_molecular_weight"
    ]
    new_columns += ["mutated_molecular_weight", "mutated_aromaticity", "mutated_isoelectric_point", "mutated_helix_fraction",
                    "mutated_turn_fraction", "mutated_sheet_fraction", "mutated_molar_extinction_1", "mutated_molar_extinction_2", "mutated_gravy"]
    new_columns += ["mutation_molecular_weight", "mutation_aromaticity", "mutation_isoelectric_point", "mutation_helix_fraction",
                    "mutation_turn_fraction", "mutation_sheet_fraction", "mutation_molar_extinction_1", "mutation_molar_extinction_2", "mutation_gravy"]
    new_columns += ["blosum62", "blosum80", "blosum90", "blosum100"]

new_columns_df = pd.DataFrame(columns=new_columns)
df = pd.concat([df, new_columns_df], axis=1)
# Split df by uniprot
unique_uniprot = df.uniprot.unique()
all_uniprot_dfs = [df[df.uniprot.eq(uniprot)].copy()
                   for uniprot in unique_uniprot]

# only one uniprot in submission case
if SUBMISSION and (len(all_uniprot_dfs) == 1):
    uniprot_df = all_uniprot_dfs[0].copy()
    if len(uniprot_df) > MAX_SUBDF_SIZE:
        # in the case of a protein with a lot of related mutation we split it into THREADS subdf
        number_splits = min(THREADS, len(uniprot_df) //
                            MAX_SUBDF_SIZE)
        all_uniprot_dfs = [uniprot_df.iloc[index, :].copy()
                           for index in np.array_split(range(len(uniprot_df)), number_splits)]
        print(
            f"divided AF70 into {len(all_uniprot_dfs)} subdf")


def compute_features(df):
    print(f"computing {df.uniprot.unique()}")
    if CONVERT_MUTATION_TO_INT:
        df["wild_aa_int"] = df["wild_aa"].apply(lambda x: aa_char2int[x])
        df["mutated_aa_int"] = df["mutated_aa"].apply(lambda x: aa_char2int[x])

    if ADD_STRUCTURE_INFOS:
        # add residue depth, sasa and c_alpha depth computed from alphafold pdb file => compute_sasa = True, compute_depth = True
        # add residue dssp infos (rsa etc.) => compute_dssp = True
        df = add_structure_infos(df, compute_sasa=True,
                                 compute_depth=True, compute_dssp=True, compute_bfactor=True,
                                 multiprocessing=True)

    if ADD_ROSETTA_SCORES:
        df = add_rosetta_scores(df, multiprocessing=True)

    if ADD_PROTEIN_ANALYSIS:
        df = add_protein_analysis(df, multiprocessing=True)

    if ADD_DEMASK_PREDICTIONS:
        df = add_demask_predictions(df, multiprocessing=True)

    return df


with Pool(THREADS) as pool:
    result_dfs = list(pool.imap(compute_features, all_uniprot_dfs))


# Merge all df together
print(len(result_dfs))
print(result_dfs[0].isna().sum().to_dict())

main_df = pd.concat(result_dfs)

print(len(main_df))
print({k: v for k, v in main_df.isna().sum().to_dict().items() if v > 0})
print(main_df.head())

if SAVE_NEW_CSV:
    main_df.to_csv(DATASET_OUTPUT_PATH, index=False)
