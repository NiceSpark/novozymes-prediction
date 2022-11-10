from Bio.PDB.ResidueDepth import ResidueDepth
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np
import pandas as pd
import tqdm
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP, ss_to_index
from biopandas.pdb import PandasPdb
from blosum import BLOSUM

DSSP_Data_Keys = [
    "DSSP index",
    "Amino acid",
    "Secondary structure",
    "Relative ASA",
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


def add_structure_infos_by_pdb(df: pd.DataFrame, alphafold_path: str,
                               compute_sasa: bool, compute_depth: bool, compute_dssp: bool, compute_bfactor: bool):
    """
    add structural information and informations computed from it
    to all rows of a df containing the pdb id pass as argument
    """

    if not alphafold_path:
        print(f"no pdb_path found for {alphafold_path}")
        return df
    pdb_parser = PDBParser()
    structure = pdb_parser.get_structure("", alphafold_path)

    # we use dicts to be able to use get() method (no check needed)
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
                       _dssp_data in enumerate(get_dssp_data(alphafold_path, structure))}
    if compute_bfactor:
        pdb_df = PandasPdb().read_pdb(alphafold_path)
        atom_df = pdb_df.df['ATOM']
        b_factor = atom_df.groupby("residue_number")[
            "b_factor"].apply(lambda x: x.median())
        pos_to_bfactor = {i: _bfactor for i,
                          _bfactor in enumerate(b_factor.to_list())}

    def add_infos_to_row(row, alphafold_path, pos_to_sasa,
                         pos_to_residue_depth, pos_to_c_alpha_depth,
                         pos_to_dssp, pos_to_bfactor):
        # we only add infos to the row with the right pdb & uniprot
        if row["alphafold_path"] == alphafold_path:
            pos = int(row["mutation_position"])
            if pos >= int(row["length"]):
                pos = int(row["length"])-1
            if pos_to_sasa:
                row["sasa"] = pos_to_sasa.get(pos, 0.0)
            if pos_to_residue_depth and pos_to_c_alpha_depth:
                row["residue_depth"] = pos_to_residue_depth.get(pos, 0.0)
                row["c_alpha_depth"] = pos_to_c_alpha_depth.get(pos, 0.0)
            if pos_to_dssp:
                for k, v in pos_to_dssp.get(pos, {}).items():
                    row[k] = v
            if pos_to_bfactor:
                row["bfactor"] = pos_to_bfactor.get(pos, 0.0)

        return row

    df = df.apply(lambda row: add_infos_to_row(row, alphafold_path, pos_to_sasa,
                                               pos_to_residue_depth, pos_to_c_alpha_depth,
                                               pos_to_dssp, pos_to_bfactor), axis=1)

    return df


def add_structure_infos(df: pd.DataFrame, compute_sasa=True, compute_depth=True, compute_dssp=True, compute_bfactor=True):
    # We want to load each structure pdb files only once, therefore we need
    # to go through each protein then each mutation

    for alphafold_path in tqdm.tqdm(df.alphafold_path.unique()):
        # 1st we get the corresponding pdb_id

        # 2nd we add the structure infos for this alphafold path:
        df = add_structure_infos_by_pdb(df, alphafold_path,
                                        compute_sasa, compute_depth, compute_dssp, compute_bfactor)

    if compute_dssp:
        # translate the Secondary Structure sign(-: None, H: AlphaHelix etc.) to a number
        df["Secondary structure"] = df["Secondary structure"].apply(lambda x:
                                                                    DSSP_codes_Secondary_Structure.get(x, {}).get("value"))

    return df


def add_protein_analysis(df):
    # fill missing pH with 7.0
    df["pH"] = df["pH"].fillna(7.0)
    tmp_sequence = df["sequence"].apply(lambda x: x.replace('-', ''))

    df["molWeight"] = tmp_sequence.apply(
        lambda x: ProteinAnalysis(x).molecular_weight())
    df["instability_index"] = tmp_sequence.apply(
        lambda x: ProteinAnalysis(x).instability_index())
    df["aromaticity"] = tmp_sequence.apply(
        lambda x: ProteinAnalysis(x).aromaticity())
    df["isoelectric_point"] = tmp_sequence.apply(
        lambda x: ProteinAnalysis(x).isoelectric_point())
    df["helix_fraction"] = tmp_sequence.apply(
        lambda x: ProteinAnalysis(x).secondary_structure_fraction()[0])
    df["turn_fraction"] = tmp_sequence.apply(
        lambda x: ProteinAnalysis(x).secondary_structure_fraction()[1])
    df["sheet_fraction"] = tmp_sequence.apply(
        lambda x: ProteinAnalysis(x).secondary_structure_fraction()[2])
    df["molar_extinction_1"] = tmp_sequence.apply(
        lambda x: ProteinAnalysis(x).molar_extinction_coefficient()[0])
    df["molar_extinction_2"] = tmp_sequence.apply(
        lambda x: ProteinAnalysis(x).molar_extinction_coefficient()[1])
    df["gravy"] = tmp_sequence.apply(
        lambda x: ProteinAnalysis(x).gravy())
    df["flexibility"] = tmp_sequence.apply(
        lambda x: sum(ProteinAnalysis(x).flexibility())/len(x))

    added_columns = ["charge_at_pH", "delta_molecular_weight", "delta_aromaticity", "delta_isoelectric_point", "delta_helix_fraction",
                     "delta_turn_fraction", "delta_sheet_fraction", "delta_molar_extinction_1", "delta_molar_extinction_2", "delta_gravy",
                     "blosum62", "blosum80", "blosum90"]
    for column in added_columns:
        df[column] = np.nan

    def add_more_protein_analysis(row):
        pos = int(row["mutation_position"])
        sequence = row["sequence"]
        mutated_sequence = sequence[:pos] + \
            row["mutated_aa"] + sequence[pos + 1:]
        mutated_sequence = mutated_sequence.replace('-', '')
        wildtype_analysis = ProteinAnalysis(sequence)
        mutated_analysis = ProteinAnalysis(mutated_sequence)

        pH = row["pH"] if row["pH"] else 7.0
        row["charge_at_pH"] = wildtype_analysis.charge_at_pH(pH)

        # deltas:
        row["delta_molecular_weight"] = (
            wildtype_analysis.molecular_weight()-mutated_analysis.molecular_weight())
        row["delta_instability_index"] = (wildtype_analysis.instability_index()
                                          - mutated_analysis.instability_index())
        row["delta_aromaticity"] = (
            wildtype_analysis.aromaticity()-mutated_analysis.aromaticity())
        row["delta_isoelectric_point"] = (
            wildtype_analysis.isoelectric_point()-mutated_analysis.isoelectric_point())
        row["delta_helix_fraction"] = (wildtype_analysis.secondary_structure_fraction()[
            0]-mutated_analysis.secondary_structure_fraction()[0])
        row["delta_turn_fraction"] = (wildtype_analysis.secondary_structure_fraction()[
            1]-mutated_analysis.secondary_structure_fraction()[1])
        row["delta_sheet_fraction"] = (wildtype_analysis.secondary_structure_fraction()[
            2]-mutated_analysis.secondary_structure_fraction()[2])
        row["delta_molar_extinction_1"] = (wildtype_analysis.molar_extinction_coefficient()[
            0]-mutated_analysis.molar_extinction_coefficient()[0])
        row["delta_molar_extinction_2"] = (wildtype_analysis.molar_extinction_coefficient()[
            1]-mutated_analysis.molar_extinction_coefficient()[1])
        row["delta_gravy"] = (wildtype_analysis.gravy() -
                              mutated_analysis.gravy())
        row["delta_flexibility"] = (sum(wildtype_analysis.flexibility())
                                    - sum(mutated_analysis.flexibility()))
        row["delta_charge_at_pH"] = (wildtype_analysis.charge_at_pH(pH)
                                     - mutated_analysis.charge_at_pH(pH))

        # add blosum 62, 80 and 90 scores for the mutation
        if row["mutated_aa"] != '-':
            mutation = row["mutated_aa"]+row["wild_aa"]
            row["blosum62"] = BLOSUM(62)[mutation]
            row["blosum80"] = BLOSUM(80)[mutation]
            row["blosum90"] = BLOSUM(90)[mutation]
        else:
            row["blosum62"] = 0.0
            row["blosum80"] = 0.0
            row["blosum90"] = 0.0

        return row

    df = df.apply(add_more_protein_analysis, axis=1)

    return df


def add_demask_predictions_by_uniprot(df: pd.DataFrame, uniprot_id: str):
    prediction_path = f"./data/main_dataset_creation/DeMaSk_outputs/predictions/{uniprot_id}.txt"
    demask_df = pd.read_csv(prediction_path, sep='\t')

    def add_infos(row, uniprot_id):
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
                print("error: prediction contains more than one element")
                print("row: ", row)
                return row

            row["demask_score"] = prediction["score"].iloc[0]
            row["demask_entropy"] = prediction["entropy"].iloc[0]
            row["demask_log2f_var"] = prediction["log2f_var"].iloc[0]
            row["demask_matrix"] = prediction["matrix"].iloc[0]
        return row

    df = df.apply(lambda row: add_infos(row, uniprot_id), axis=1)
    return df


def add_demask_predictions(df: pd.DataFrame):
    # we add the columns to the df
    for column in ["demask_score", "demask_entropy", "demask_log2f_var", "demask_matrix"]:
        df[column] = 0.0

    for uniprot_id in tqdm.tqdm(df.uniprot.unique()):
        df = add_demask_predictions_by_uniprot(df, uniprot_id)

    return df
