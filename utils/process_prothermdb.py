##### ProThermDB #####
import pandas as pd
import numpy as np

COLUMNS = ["PDB_wild", "uniprot", "mutated_chain",
           "mutation_code", "pH", "Texp", "Tm", "ddG", "dTm", "flag_for_remove"]
# "mutation_sequence_code"
# turn to False if you want to save all available info in the db
SAVE_ONLY_COLUMNS = True


def add_missing_column(df):
    for name in COLUMNS:
        if name not in df.columns.to_list():
            df[name] = np.nan

    return df


all_ddg_df = pd.read_csv("./data/ProThermDB/prothermdb_all_ddg.tsv", sep="\t")
all_dtm_df = pd.read_csv("./data/ProThermDB/prothermdb_all_dTm.tsv", sep="\t")
all_tm_df = pd.read_csv("./data/ProThermDB/prothermdb_all_Tm.tsv", sep="\t")

prothermdb_df = pd.concat(
    [all_ddg_df, all_dtm_df, all_tm_df], ignore_index=True)

# removes duplicate
prothermdb_df = prothermdb_df.drop_duplicates()

# columns we let "as is"

prothermdb_df.rename(columns={"PROTEIN": "protein", "UniProt_ID": "uniprot", "PubMed_ID": "PMID",
                              "SOURCE": "source", 'T_(C)': "Texp", 'MEASURE': "measure", 'METHOD': "method",
                              '∆G_(kcal/mol)': "dG", '∆∆G_(kcal/mol)': "ddG", 'Tm_(C)': "Tm", '∆Tm_(C)': "dTm"},
                     inplace=True)

prothermdb_df = add_missing_column(prothermdb_df)


def process_prothermdb(row):
    try:
        for k in row.keys():
            if row[k] == '-':
                row[k] = ""

        # coherent PDB_WILD: all caps
        row["PDB_wild"] = row["PDB_wild"].upper()

        # 1st: convert T to float (sometimes: 23(1.2) or >96)
        if '>' in row["Tm"]:
            row["Tm"] = row["Tm"].split(">")[1]

        texp = float(row["Texp"].split('(')[0]) if row["Texp"] else np.nan
        tm = float(row["Tm"].split('(')[0]) if row["Tm"] else np.nan
        # 2nd: convert temperatures from C to K
        row["Texp"] = texp + 273.15 if texp else np.nan
        row["Tm"] = tm + 273.15 if tm else np.nan

        # make sure that dTm, ddG and Tm are all floats:
        row["ddG"] = float(str(row["ddG"]).split(
            '(')[0]) if row["ddG"] else np.nan
        row["dTm"] = float(str(row["dTm"]).split(
            '(')[0]) if row["dTm"] else np.nan
        row["Tm"] = float(str(row["Tm"]).split(
            '(')[0]) if row["Tm"] else np.nan

        # handling the difference in mutation code, including multiple mutations
        row["mutated_chain"] = ""
        row["mutation_code"] = ""

        if row["PDB_Chain_Mutation"]:
            pdb_split = [s.split(":")
                         for s in row["PDB_Chain_Mutation"].split(' ')]
            for s in pdb_split:
                if len(s) != 2:
                    continue
                [pdb_wild, mut] = s
                # normally we have 1csp_A => A
                if '_' in pdb_wild:
                    row["mutated_chain"] += pdb_wild.split('_')[-1]

                # sometimes we have A_M1R => A & M1R
                if '_' in mut:
                    row["mutated_chain"] += mut.split('_')[0]
                    row["mutation_code"] += mut.split('_')[1]+" "
                else:
                    row["mutation_code"] += mut+" "

        # M1R E3K K65I E66K(Based on UniProt and PDB) => M1R E3K K65I E66K
        row["mutation_sequence_code"] = row["MUTATION"].split('(')[0]

        # we only need first char of mutated_chain
        row["mutated_chain"] = row["mutated_chain"][0] if row["mutated_chain"] else ""
        # remove last space in str
        row["mutation_code"] = row["mutation_code"].strip()
        row["mutation_sequence_code"] = row["mutation_sequence_code"].strip()
        # remove spaces in some uniprot id:
        row["uniprot"] = row["uniprot"].replace(" ", "")

        # remove empty/multiple mutation code
        if type(row["mutation_code"]) != type(""):
            row["flag_for_remove"] = True
        elif len(row["mutation_code"]) < 2:
            row["flag_for_remove"] = True
        elif ' ' in row["mutation_code"]:
            row["flag_for_remove"] = True

        return row
    except:
        row["flag_for_remove"] = True
        return row


prothermdb_df = prothermdb_df.apply(process_prothermdb, axis=1)

prothermdb_df.drop(columns=['NO', 'KEY_WORDS', 'REFERENCE', 'AUTHOR', 'REMARKS', 'RELATED_ENTRIES', "MUTATION", "PDB_Chain_Mutation"],
                   inplace=True)
prothermdb_df = prothermdb_df[~prothermdb_df.flag_for_remove.eq(True)]

prothermdb_df = prothermdb_df[COLUMNS +
                              (list(set(prothermdb_df.columns.to_list())-set(COLUMNS)))]
prothermdb_df.to_csv(
    f"./data/ProThermDB/processed_prothermdb.csv", index=False)
