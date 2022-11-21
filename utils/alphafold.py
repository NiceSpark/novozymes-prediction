import urllib.request
import glob
import tqdm
from biopandas.pdb import PandasPdb

from .infos_translation import aa_tri2char


def download_alphafold(alphafold_id):
    """
    download alphafold prediction corresponding to the alphafold id in the row of data
    if there is already a corresponding pdb we do not download it again
    """
    DIR = "./data/main_dataset_creation/3D_structures/alphafold"

    url = f"https://alphafold.ebi.ac.uk/files/AF-{alphafold_id}-F1-model_v3.pdb"
    path = f"{DIR}/{alphafold_id}.pdb"

    if glob.glob(path):
        return path

    try:
        urllib.request.urlretrieve(url, path)
        return path
    except Exception as e:
        print(f"exception raised for alphafold_id: {alphafold_id}:\n{e}")
        return ""


def check_atom_coherence_by_row(sequence, alphafold_path):
    try:
        atom_df = PandasPdb().read_pdb(alphafold_path)
    except Exception as e:
        print(f"error for {alphafold_path}: Exception: {e}")
        return False
    atom_df = atom_df.df['ATOM']
    seq_df = atom_df[["residue_name", "residue_number"]].drop_duplicates()
    if len(seq_df) != len(sequence):
        print(f"error for {alphafold_path}: sequences length don't match")
        return False
    for i, c in enumerate(seq_df.residue_name):
        if sequence[i] != aa_tri2char[c]:
            print(
                f"error for {alphafold_path} at position {i}: {aa_tri2char[c]} instead of {sequence[i]}")
            return False
    return True


def check_atom_coherence(df):
    count = 0
    unique_df = df[["sequence", "alphafold_path"]]
    unique_df = unique_df[~(unique_df.alphafold_path.isna())].drop_duplicates()
    print(
        f"checking coherence between {len(unique_df)} pairs of sequence-atom(pdb) files")
    non_coherent_pdb_files = []
    for _, row in tqdm.tqdm(unique_df.iterrows()):
        if not check_atom_coherence_by_row(row.sequence, row.alphafold_path):
            count += 1
            non_coherent_pdb_files.append(row.alphafold_path)

    print(f"found {count} non coherent sequence-atom(pdb) pairs")

    coherent_pdb = df[~(df.alphafold_path.isna())]
    coherent_pdb = coherent_pdb[~(
        coherent_pdb.alphafold_path.isin(non_coherent_pdb_files))]
    return coherent_pdb
