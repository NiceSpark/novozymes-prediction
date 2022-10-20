import pandas as pd

def get_uniprot_infos(pdb_ids):
    """
    Get uniprotkdb infos for each pdb wildtype id contain in a list.
    Infos contains:
    - uniprot,
    - sequence,length,molWeight,
    - countByFeatureType,chain_start,chain_end,
    - AlphaFoldDB
    """
    results = {}

    pdb_uniprot_mapping_df = pd.read_csv("./data/main_dataset/pdb_uniprot_mapping.csv")
    uniprot_infos_df = pd.read_csv("./data/main_dataset/uniprot_infos.csv")

    uniprot_ids = pdb_uniprot_mapping_df[pdb_uniprot_mapping_df["PDB_wild"].isin(pdb_ids)]
    
    mapping = list(map(lambda x: [x["PDB_wild"], x["uniprot"]], uniprot_ids.T.to_dict().values()))
    uniprot_infos = uniprot_infos_df[uniprot_infos_df["uniprot"].isin([x[1] for x in mapping])]

    for [pdb, uniprot] in mapping:
        results = dict(results, 
                        **{pdb: uniprot_infos.loc[uniprot_infos.uniprot.eq(uniprot)].to_dict("records")[0]}
        )

    return results