import glob
import tqdm
import os

# This script is to be used on the distant machine
# path is therefor relative (put this script in DeMaSk_outputs/)

fasta_dir = "./fasta"
homologs_dir = "./homologs"
predictions_dir = "./predictions"

fasta_files = glob.glob(fasta_dir+"/*.fa")
homologs_files = glob.glob(homologs_dir+"/*.a2m")
predictions_files = glob.glob(predictions_dir+"/*.txt")
print(f"found {len(fasta_files)} fasta files")
print(f"found {len(homologs_files)} homologs_files")
print(f"found {len(predictions_files)} predictions_files")

# create homologs files
for fasta_path in tqdm.tqdm(fasta_files):
    uniprot, _ = os.path.splitext(fasta_path.split('/')[-1])
    homologs_path = f"{homologs_dir}/{uniprot}.a2m"
    # check that homologs_path does not exist yet:
    if glob.glob(homologs_path):
        print(f"homologs already found for {uniprot}")
        continue

    cmd = f"python3 -m demask.homologs -s {fasta_path} -o {homologs_path}"
    os.system(cmd)


# create prediction files
homologs_files = glob.glob(homologs_dir+"/*.a2m")
print(f"found {len(homologs_files)} homologs_files")
for homologs_path in tqdm.tqdm(homologs_files):
    uniprot, _ = os.path.splitext(homologs_path.split('/')[-1])
    predictions_path = f"{predictions_dir}/{uniprot}.txt"
    # check that predictions_path does not exist yet:
    if glob.glob(predictions_path):
        print(f"prediction already found for {uniprot}")
        continue

    cmd = f"python3 -m demask.predict -i {homologs_path} -o {predictions_path}"
    os.system(cmd)
