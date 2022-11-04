import urllib.request
import glob


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
