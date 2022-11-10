import os
import pandas as pd

ROSETTA_BIN_DIR = "/home/ml/novozymes-prediction/resources/rosetta/rosetta_bin_linux_2021.16.61629_bundle/main/source/bin/"
RELAX_BIN = f"{ROSETTA_BIN_DIR}relax.static.linuxgccrelease"


# MAX_CYCLES = 10000
# -default_max_cycles {MAX_CYCLES}
ROSETTA_PARAMETERS = f"-relax:constrain_relax_to_start_coords -out:suffix _relaxed -out:no_nstruct_label -relax:ramp_constraints false"

DATASET_PATH = "../data/main_dataset_creation/outputs/all_v2/dataset_with_alphafold_paths.csv"

df = pd.read_csv(DATASET_PATH)
alphafold_paths = df.alphafold_path.unique()


for alphafold_path in alphafold_paths:
    cmd = f"{RELAX_BIN} -in:file:s .{alphafold_path} {ROSETTA_PARAMETERS}"
    # print(cmd)
    os.system(cmd)
    name, _ = os.path.splitext(alphafold_path.split("/")[-1])
    if os.path.exists("./score_relaxed.sc"):
        os.system(f"mv score_relaxed.sc {name}_score_relaxed.sc")
