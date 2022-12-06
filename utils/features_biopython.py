from Bio.PDB.ResidueDepth import ResidueDepth
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np
import pandas as pd
import os
import warnings
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.DSSP import DSSP, ss_to_index
from biopandas.pdb import PandasPdb
from blosum import BLOSUM
from .file_utils import open_json, write_json

SUBSET_DUPLICATES_NO_PH = ["uniprot", "wild_aa", "mutation_position",
                           "mutated_aa", "sequence"]

BLOSUM100 = BLOSUM(
    "/home/ml/novozymes-prediction/data/main_dataset_creation/blosum/blosum100.mat")

DSSP_Data_Keys = [
    "DSSP index",
    "Amino acid",
    "Secondary_structure",
    "Relative_ASA",
    "Phi",
    "Psi",
    "NH->O_1_relidx",
    "NH->O_1_energy",
    "O->NH_1_relidx",
    "O->NH_1_energy",
    "NH->O_2_relidx",
    "NH->O_2_energy",
    "O->NH_2_relidx",
    "O->NH_2_energy"
]

DSSP_codes_Secondary_Structure = {
    "H": {"value": 0, "name": "Alpha helix (4-12)"},
    "B": {"value": 1, "name": "Isolated beta-bridge residue"},
    "E": {"value": 2, "name": "Strand"},
    "G": {"value": 3, "name": "3-10 helix"},
    "I": {"value": 4, "name": "Pi helix"},
    "T": {"value": 5, "name": "Turn"},
    "S": {"value": 6, "name": "Bend"},
    "-": {"value": 7, "name": "None"}
}


def get_depth_maps(structure):
    # Get residue depth model and use to calculate rd and ca depth
    model = structure[0]
    residue_depth = ResidueDepth(model)
    residue_depth, c_alpha_depth = list(
        zip(*[(x[1][0], x[1][1]) for x in residue_depth]))

    # Create mappings
    return {i: x for i, x in enumerate(residue_depth)}, {i: x for i, x in enumerate(c_alpha_depth)}


def get_sasa(structure, sr_n_points=250):
    """
    get residues Solvent Accessible Surface Areas (sasa)
    Calculates SASAs using the Shrake-Rupley algorithm.
    """

    sr = ShrakeRupley(n_points=sr_n_points)
    sr.compute(structure, level="R")
    return [x.sasa for x in structure.get_residues()]


def get_dssp_data(alphafold_path: str, structure):
    model = structure[0]
    dssp = DSSP(model, alphafold_path)

    # result is a list containing len(sequence) elements
    # each elements contain the DSSP data (Secondary Structure, RSA, etc.) for each residue
    result = [dict(list(zip(DSSP_Data_Keys, list(dssp_val)))[2:])
              for dssp_val in list(dssp)]
    return result


def get_structure_infos(structure_path: str, compute_sasa: bool, compute_depth: bool,
                        compute_dssp: bool, compute_bfactor: bool):
    # we use dicts to be able to use get() method (no check needed)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', PDBConstructionWarning)
        pdb_parser = PDBParser()
        structure = pdb_parser.get_structure("", structure_path)
    pos_to_sasa = {}
    pos_to_residue_depth, pos_to_c_alpha_depth = {}, {}
    pos_to_dssp = {}
    pos_to_bfactor = {}

    if compute_sasa:
        sasa_list = get_sasa(structure)
        pos_to_sasa = {i: _sasa for i, _sasa in enumerate(sasa_list)}
    if compute_depth:
        pos_to_residue_depth, pos_to_c_alpha_depth = get_depth_maps(structure)
    if compute_dssp:
        pos_to_dssp = {i: _dssp_data for i,
                       _dssp_data in enumerate(get_dssp_data(structure_path, structure))}
    if compute_bfactor:
        pdb_df = PandasPdb().read_pdb(structure_path)
        atom_df = pdb_df.df['ATOM']
        b_factor = atom_df.groupby("residue_number")[
            "b_factor"].apply(lambda x: x.median())
        pos_to_bfactor = {i: _bfactor for i,
                          _bfactor in enumerate(b_factor.to_list())}

    return [pos_to_sasa, pos_to_residue_depth, pos_to_c_alpha_depth, pos_to_dssp, pos_to_bfactor]


def add_structure_infos_by_protein(df: pd.DataFrame, structure_path: str, prefix: str,
                                   compute_sasa: bool, compute_depth: bool, compute_dssp: bool,
                                   compute_bfactor: bool):
    """
    add structural information and informations computed from it
    to all rows of a df containing the pdb id pass as argument
    """

    if not structure_path:
        print(f"no pdb_path found for {structure_path}")
        return df

    name, _ = os.path.splitext(structure_path.split("/")[-1])
    name = name.replace("_relaxed", '')
    infos = get_structure_infos(
        structure_path, compute_sasa, compute_depth, compute_dssp, compute_bfactor)
    [pos_to_sasa, pos_to_residue_depth, pos_to_c_alpha_depth,
        pos_to_dssp, pos_to_bfactor] = infos

    def add_infos_to_row(row, prefix, name, pos_to_sasa,
                         pos_to_residue_depth, pos_to_c_alpha_depth,
                         pos_to_dssp, pos_to_bfactor):
        # we only add infos to the row with the right alphafold_path
        if row["alphafold_path"].split('/')[-1] == f"{name}.pdb":
            pos = int(row["mutation_position"])
            if pos >= int(row["length"]):
                pos = int(row["length"])-1
            if pos_to_sasa:
                row[f"{prefix}_sasa"] = pos_to_sasa.get(pos, np.nan)
            if pos_to_residue_depth and pos_to_c_alpha_depth:
                row[f"{prefix}_residue_depth"] = pos_to_residue_depth.get(
                    pos, np.nan)
                row[f"{prefix}_c_alpha_depth"] = pos_to_c_alpha_depth.get(
                    pos, np.nan)
            if pos_to_dssp:
                for k, v in pos_to_dssp.get(pos, {}).items():
                    row[f"{prefix}_{k}"] = v
            if pos_to_bfactor:
                row[f"{prefix}_bfactor"] = pos_to_bfactor.get(pos, "error")

        return row

    df = df.apply(lambda row: add_infos_to_row(row, prefix, name, pos_to_sasa,
                                               pos_to_residue_depth, pos_to_c_alpha_depth,
                                               pos_to_dssp, pos_to_bfactor), axis=1)

    return df


def add_structure_infos_by_mutation(mutations_df: pd.DataFrame,
                                    compute_sasa: bool, compute_depth: bool, compute_dssp: bool,
                                    compute_bfactor: bool):
    """
    add structural information and informations computed from it
    to all rows of a df containing the pdb id pass as argument
    """

    def add_infos_to_row(row, errors):
        # we get the infos for this mutation
        w_aa, m_aa = row["wild_aa"], row["mutated_aa"]
        relaxed_pos = int(row["mutation_position"])+1
        name, _ = os.path.splitext(row["alphafold_path"].split("/")[-1])
        mutated_structure_path = row["relaxed_mutated_3D_path"]
        infos = get_structure_infos(
            mutated_structure_path, compute_sasa, compute_depth, compute_dssp, compute_bfactor)
        [pos_to_sasa, pos_to_residue_depth, pos_to_c_alpha_depth,
            pos_to_dssp, pos_to_bfactor] = infos

        # we add the infos
        pos = int(row["mutation_position"])
        if pos >= int(row["length"]):
            pos = int(row["length"])-1
        if pos_to_sasa:
            row["mutated_relaxed_sasa"] = pos_to_sasa.get(pos, np.nan)
        if pos_to_residue_depth and pos_to_c_alpha_depth:
            row["mutated_relaxed_residue_depth"] = pos_to_residue_depth.get(
                pos, np.nan)
            row["mutated_relaxed_c_alpha_depth"] = pos_to_c_alpha_depth.get(
                pos, np.nan)
        if pos_to_dssp:
            for k, v in pos_to_dssp.get(pos, {}).items():
                row[f"mutated_relaxed_{k}"] = v
        if pos_to_bfactor:
            row["mutated_relaxed_bfactor"] = pos_to_bfactor.get(pos, np.nan)

        errors.append(
            {f"row": row})

        with open("add_structure_infos_by_mutation.log", "a+") as f:
            f.write(f"computed: {mutated_structure_path} \n")

        return row

    errors = []
    mutations_df = mutations_df.apply(
        lambda row: add_infos_to_row(row, errors), axis=1)

    return mutations_df


def update_main_df(row, infos_df, new_columns):
    # SUBSET_DUPLICATES_NO_PH = ["uniprot", "wild_aa", "mutation_position",
    #                            "mutated_aa", "sequence"]
    associated_row = infos_df.loc[
        (infos_df["uniprot"] == row["uniprot"]) &
        (infos_df["wild_aa"] == row["wild_aa"]) &
        (infos_df["mutation_position"] == row["mutation_position"]) &
        (infos_df["mutated_aa"] == row["mutated_aa"]) &
        (infos_df["sequence"] == row["sequence"])
    ]
    if len(associated_row) == 1:
        associated_embeddings_row = associated_row.iloc[0, :]
        for col in new_columns:
            row[col] = associated_embeddings_row[col]

    return row


def add_structure_infos(df: pd.DataFrame, compute_sasa=True, compute_depth=True, compute_dssp=True, compute_bfactor=True, multiprocessing=False):
    # We want to load each structure pdb files only once, therefore we need
    # to go through each protein then each mutation
    new_columns = []
    for prefix in ["alphafold", "wild_relaxed", "mutated_relaxed", "mutation"]:
        new_columns += [f"{prefix}_{k}" for k in DSSP_Data_Keys[2:]]
        new_columns += [f"{prefix}_{k}" for k in ["sasa", "residue_depth",
                                                  "c_alpha_depth", "bfactor"]]
    if not multiprocessing:
        for col in new_columns:
            df[col] = np.nan

    for alphafold_path in df.alphafold_path.unique():
        name, _ = os.path.splitext(alphafold_path.split("/")[-1])

        wild_relaxed_path = f"./compute_mutated_structures/relaxed_pdb/{name}_relaxed/{name}_relaxed.pdb"
        try:
            df = add_structure_infos_by_protein(df, alphafold_path, "alphafold",
                                                compute_sasa, compute_depth, compute_dssp, compute_bfactor)
        except Exception as e:
            errors = open_json(
                "/home/jupyter/novozymes-prediction/errors.json")
            errors.append(
                f"error happend for {name} in add_structure_infos_by_protein (alphafold): {e}")
        try:
            df = add_structure_infos_by_protein(df, wild_relaxed_path, "wild_relaxed",
                                                compute_sasa, compute_depth, compute_dssp, compute_bfactor)
        except Exception as e:
            errors = open_json(
                "/home/jupyter/novozymes-prediction/errors.json")
            errors.append(
                f"error happend for {name} in add_structure_infos_by_protein (wild_relaxed): {e}")

    unique_mutations_df = df.copy()
    unique_mutations_df.drop_duplicates(
        subset=SUBSET_DUPLICATES_NO_PH, inplace=True)
    try:
        unique_mutations_df = add_structure_infos_by_mutation(unique_mutations_df,
                                                              compute_sasa, compute_depth, compute_dssp,
                                                              compute_bfactor)
    except Exception as e:
        errors = open_json("/home/jupyter/novozymes-prediction/errors.json")
        errors.append(
            f"error happend for {name} in add_structure_infos_by_mutation: {e}")

    # add deltas
    unique_mutations_df["mutation_sasa"] = unique_mutations_df["mutated_relaxed_sasa"] - \
        unique_mutations_df["wild_relaxed_sasa"]
    unique_mutations_df["mutation_residue_depth"] = unique_mutations_df["mutated_relaxed_residue_depth"] - \
        unique_mutations_df["wild_relaxed_residue_depth"]
    unique_mutations_df["mutation_c_alpha_depth"] = unique_mutations_df["mutated_relaxed_c_alpha_depth"] - \
        unique_mutations_df["wild_relaxed_c_alpha_depth"]
    for k in DSSP_Data_Keys[2:]:
        if k == DSSP_Data_Keys[2]:
            # does not make sense to make a difference on the structure (same for everyone)
            unique_mutations_df[f"mutation_{k}"] = unique_mutations_df[f"mutated_relaxed_{k}"]
        else:
            unique_mutations_df[f"mutation_{k}"] = unique_mutations_df[f"mutated_relaxed_{k}"] - \
                unique_mutations_df[f"wild_relaxed_{k}"]
    unique_mutations_df["mutation_bfactor"] = unique_mutations_df["mutated_relaxed_bfactor"] - \
        unique_mutations_df["wild_relaxed_bfactor"]

    df = df.apply(lambda row: update_main_df(
        row, unique_mutations_df, new_columns), axis=1)

    for prefix in ["alphafold", "wild_relaxed", "mutated_relaxed", "mutation"]:
        if compute_dssp:
            # translate the Secondary Structure sign(-: None, H: AlphaHelix etc.) to a number
            df[f"{prefix}_Secondary_structure"] = df[f"{prefix}_Secondary_structure"].apply(lambda x:
                                                                                            DSSP_codes_Secondary_Structure.get(x, {}).get("value", np.nan))

    return df


def add_protein_analysis(df, multiprocessing=False):
    if not multiprocessing:
        new_columns = [
            "wild_charge_at_pH",
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
        for column in new_columns:
            df[column] = np.nan

    wild_sequence = df["sequence"].apply(lambda x: x.replace('-', ''))

    df["wild_molecular_weight"] = wild_sequence.apply(
        lambda x: ProteinAnalysis(x).molecular_weight())
    df["wild_instability_index"] = wild_sequence.apply(
        lambda x: ProteinAnalysis(x).instability_index())
    df["wild_aromaticity"] = wild_sequence.apply(
        lambda x: ProteinAnalysis(x).aromaticity())
    df["wild_isoelectric_point"] = wild_sequence.apply(
        lambda x: ProteinAnalysis(x).isoelectric_point())
    df["wild_helix_fraction"] = wild_sequence.apply(
        lambda x: ProteinAnalysis(x).secondary_structure_fraction()[0])
    df["wild_turn_fraction"] = wild_sequence.apply(
        lambda x: ProteinAnalysis(x).secondary_structure_fraction()[1])
    df["wild_sheet_fraction"] = wild_sequence.apply(
        lambda x: ProteinAnalysis(x).secondary_structure_fraction()[2])
    df["wild_molar_extinction_1"] = wild_sequence.apply(
        lambda x: ProteinAnalysis(x).molar_extinction_coefficient()[0])
    df["wild_molar_extinction_2"] = wild_sequence.apply(
        lambda x: ProteinAnalysis(x).molar_extinction_coefficient()[1])
    df["wild_gravy"] = wild_sequence.apply(
        lambda x: ProteinAnalysis(x).gravy())
    df["wild_flexibility"] = wild_sequence.apply(
        lambda x: sum(ProteinAnalysis(x).flexibility())/len(x))

    def add_more_protein_analysis(row):
        pos = int(row["mutation_position"])
        sequence = row["sequence"]
        mutated_sequence = sequence[:pos] + \
            row["mutated_aa"] + sequence[pos + 1:]
        mutated_sequence = mutated_sequence.replace('-', '')
        wildtype_analysis = ProteinAnalysis(sequence)
        mutated_analysis = ProteinAnalysis(mutated_sequence)

        row["wild_charge_at_pH"] = wildtype_analysis.charge_at_pH(row["pH"])

        # indirect:
        row["mutated_molecular_weight"] = mutated_analysis.molecular_weight()
        row["mutated_instability_index"] = mutated_analysis.instability_index()
        row["mutated_aromaticity"] = mutated_analysis.aromaticity()
        row["mutated_isoelectric_point"] = mutated_analysis.isoelectric_point()
        row["mutated_helix_fraction"] = mutated_analysis.secondary_structure_fraction()[
            0]
        row["mutated_turn_fraction"] = mutated_analysis.secondary_structure_fraction()[
            1]
        row["mutated_sheet_fraction"] = mutated_analysis.secondary_structure_fraction()[
            2]
        row["mutated_molar_extinction_1"] = mutated_analysis.molar_extinction_coefficient()[
            0]
        row["mutated_molar_extinction_2"] = mutated_analysis.molar_extinction_coefficient()[
            1]
        row["mutated_gravy"] = mutated_analysis.gravy()
        row["mutated_flexibility"] = sum(mutated_analysis.flexibility())
        row["mutated_charge_at_pH"] = mutated_analysis.charge_at_pH(row["pH"])

        # deltas:
        row["mutation_molecular_weight"] = (mutated_analysis.molecular_weight()
                                            - wildtype_analysis.molecular_weight())
        row["mutation_instability_index"] = (mutated_analysis.instability_index()
                                             - wildtype_analysis.instability_index())
        row["mutation_aromaticity"] = (mutated_analysis.aromaticity()
                                       - wildtype_analysis.aromaticity())
        row["mutation_isoelectric_point"] = (mutated_analysis.isoelectric_point()
                                             - wildtype_analysis.isoelectric_point())
        row["mutation_helix_fraction"] = (mutated_analysis.secondary_structure_fraction()[0]
                                          - wildtype_analysis.secondary_structure_fraction()[0])
        row["mutation_turn_fraction"] = (mutated_analysis.secondary_structure_fraction()[1]
                                         - wildtype_analysis.secondary_structure_fraction()[1])
        row["mutation_sheet_fraction"] = (mutated_analysis.secondary_structure_fraction()[2]
                                          - wildtype_analysis.secondary_structure_fraction()[2])
        row["mutation_molar_extinction_1"] = (mutated_analysis.molar_extinction_coefficient()[0]
                                              - wildtype_analysis.molar_extinction_coefficient()[0])
        row["mutation_molar_extinction_2"] = (mutated_analysis.molar_extinction_coefficient()[1]
                                              - wildtype_analysis.molar_extinction_coefficient()[1])
        row["mutation_gravy"] = (mutated_analysis.gravy()
                                 - wildtype_analysis.gravy())
        row["mutation_flexibility"] = (sum(mutated_analysis.flexibility())
                                       - sum(wildtype_analysis.flexibility()))
        row["mutation_charge_at_pH"] = (mutated_analysis.charge_at_pH(row["pH"])
                                        - wildtype_analysis.charge_at_pH(row["pH"]))

        # add blosum 62, 80, 90 and 100 scores for the mutation
        # NB: blosum scores give an idea of how 2 amino acid are "close" or not, therefore it is symmetric
        # ie. no need for direct and indirect blosum scores
        if row["mutated_aa"] != '-':
            mutation = row["wild_aa"]+row["mutated_aa"]
            row["blosum62"] = BLOSUM(62)[mutation]
            row["blosum80"] = BLOSUM(80)[mutation]
            row["blosum90"] = BLOSUM(90)[mutation]
            row["blosum100"] = BLOSUM100[mutation]
        else:
            row["blosum62"] = 0.0
            row["blosum80"] = 0.0
            row["blosum90"] = 0.0
            row["blosum100"] = 0.0
        return row

    df = df.apply(add_more_protein_analysis, axis=1)

    return df
