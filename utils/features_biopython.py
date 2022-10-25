from Bio.PDB.ResidueDepth import ResidueDepth
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy import stats
import pandas as pd
import numpy as np
import tqdm
import Bio
import sys
import os

pdb_parser = PDBParser(QUIET=1)


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


def add_struct_infos_by_pdb(df: pd.DataFrame, pdb_id: str, uniprot_id: str, alphafold_path: str):
    """
    add structural information and informations computed from it
    to all rows of a df containing the pdb id pass as argument
    """

    if not alphafold_path:
        print(f"no pdb_path found for {alphafold_path}")
        return df
    struct = pdb_parser.get_structure(pdb_id, alphafold_path)

    sasa_list = get_sasa(struct)

    # we use a dict to be able to use get() method (no check needed)
    pos_to_sasa = {i: _sasa for i, _sasa in enumerate(sasa_list)}
    pos_to_residue_depth, pos_to_c_alpha_depth = get_depth_maps(struct)

    def add_infos_to_row(row, pdb_id, uniprot_id, pos_to_sasa, pos_to_residue_depth, pos_to_c_alpha_depth):
        if (row["PDB_wild"] == pdb_id and row["uniprot"] == uniprot_id):
            pos = int(row["mutation_position"])
            row["sasa"] = pos_to_sasa.get(pos, 0.0)
            row["residue_depth"] = pos_to_residue_depth.get(pos, 0.0)
            row["c_alpha_depth"] = pos_to_c_alpha_depth.get(pos, 0.0)
        return row

    df = df.apply(lambda row: add_infos_to_row(row, pdb_id, uniprot_id, pos_to_sasa,
                                               pos_to_residue_depth, pos_to_c_alpha_depth), axis=1)

    return df


def add_struct_infos(df: pd.DataFrame):
    # We want to load each struct pdb files only once, therefore we need
    # to go through each protein then each mutation

    for path in tqdm.tqdm(df.alphafold_path.unique()):
        # 1st we get the corresponding pdb_id
        # for now we just take the corresponding pdb id in the 1st row with that path
        pdb_id = df.loc[df.alphafold_path.eq(path)].iloc[0, :]["PDB_wild"]
        uniprot_id = df.loc[df.alphafold_path.eq(path)].iloc[0, :]["uniprot"]

        # 2nd we add the struct infos for this alphafold path & pdb:
        df = add_struct_infos_by_pdb(df, pdb_id, uniprot_id, path)

    return df


def add_protein_analysis(df):
    df["stability_analysis"] = df["protein_sequence"].apply(
        lambda x: ProteinAnalysis(x).instability_index())
    df["aromaticity_analysis"] = df["protein_sequence"].apply(
        lambda x: ProteinAnalysis(x).aromaticity())
    df["isoelectric_analysis"] = df["protein_sequence"].apply(
        lambda x: ProteinAnalysis(x).isoelectric_point())
    df["charge_analysis"] = df["protein_sequence"].apply(
        lambda x: ProteinAnalysis(x).charge_at_pH(8.0))
    df["helix_analysis"] = df["protein_sequence"].apply(
        lambda x: ProteinAnalysis(x).secondary_structure_fraction()[0])
    df["turn_analysis"] = df["protein_sequence"].apply(
        lambda x: ProteinAnalysis(x).secondary_structure_fraction()[1])
    df["sheet_analysis"] = df["protein_sequence"].apply(
        lambda x: ProteinAnalysis(x).secondary_structure_fraction()[2])
    df["mec_analysis_1"] = df["protein_sequence"].apply(
        lambda x: ProteinAnalysis(x).molar_extinction_coefficient()[0])
    df["mec_analysis_2"] = df["protein_sequence"].apply(
        lambda x: ProteinAnalysis(x).molar_extinction_coefficient()[1])
    df["gravy_analysis"] = df["protein_sequence"].apply(
        lambda x: ProteinAnalysis(x).gravy())

    return df
