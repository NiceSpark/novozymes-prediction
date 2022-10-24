import pandas as pd
import numpy as np
import math


def get_uniprot_infos(pdb_ids=[], uniprot_ids=[], 
                        pdb_uniprot_mapping_df=None, uniprot_infos_df=None):
    """
    This function works with either pdb_ids or uniprot_ids

    Gets uniprotkdb infos for each pdb wildtype id contain in a list
    (and/or each uniprot_ids in a list)
    Infos contains:
    - uniprot,
    - sequence,length,molWeight,
    - countByFeatureType,chain_start,chain_end,
    - AlphaFoldDB

    @returns: a dict which keys are the elements of pdb_ids and uniprot_ids

    """
    results = {}
    if pdb_uniprot_mapping_df is None:
        print("reading pdb_uniprot_mapping_df")
        print(type(pdb_uniprot_mapping_df))
        pdb_uniprot_mapping_df = pd.read_csv("./data/main_dataset/pdb_uniprot_mapping.csv")
    if uniprot_infos_df is None:
        print("reading uniprot_infos_df")
        uniprot_infos_df = pd.read_csv("./data/main_dataset/uniprot_infos.csv")
    if pdb_ids:
        pdb_uniprot_ids = pdb_uniprot_mapping_df[pdb_uniprot_mapping_df["PDB_wild"].isin(pdb_ids)]
        
        mapping = list(map(lambda x: [x["PDB_wild"], x["uniprot"]], pdb_uniprot_ids.T.to_dict().values()))
        uniprot_infos = uniprot_infos_df[uniprot_infos_df["uniprot"].isin([x[1] for x in mapping])]
        for [pdb, uniprot] in mapping:
            info_lst = uniprot_infos.loc[uniprot_infos.uniprot.eq(uniprot)].to_dict("records")
            infos = info_lst[0] if len(info_lst)>0 else {}

            results = dict(results, 
                            **{pdb: infos}
            )

    if uniprot_ids:
        uniprot_infos_dict = uniprot_infos_df.to_dict("records")
        uniprot_infos_dict = {r["uniprot"]: r for r in uniprot_infos_dict}
        results = dict(results, **uniprot_infos_dict)
    return results

def convert_columns(df, with_infos=False):
    """convert columns to the right type, notably float type for pH, Tm, ddG & dTm"""
    df = df.astype({"PDB_wild": object, "mutated_chain": object, 
                    "mutation_code": object,
                    "pH": float, "Tm": float, "ddG": float, 
                    "dTm": float
                    })
    
    if with_infos:
        df = df.astype({"sequence": object, "length": object, 
                        "molWeight": object, "countByFeatureType": object, 
                        "chain_start": float, "chain_end": float, 
                        "AlphaFoldDB": object
                        })
            
    # "mutation_sequence_code": object, "Texp": float,
    return df


def correct_mutation_code(mutation_code: str, sequence: str, chain_start: float, neighbors=1):
    """
    This function check if the mutation_code is coherent
    if it's not it tries to correct it thanks to the chain_start argument

    @returns:
        the correct mutation_code
        or "" if no mutation_code is false and cannot be corrected
    """

    if (not mutation_code or type(mutation_code)!=type("")):
        return ""
    if type(sequence)!=type(""):
        return ""
    if (type(chain_start)==float and not math.isnan(chain_start)):
        chain_start = int(chain_start)
    else:
        chain_start = 0
    
    wildtype_aa = mutation_code[0]
    mutation_aa = mutation_code[-1]
    try:
        position = int(mutation_code[1:-1])
    except:
        print(f"error occured, mutation code cannot be parsed, mutation_code: {mutation_code}")
        return ""

    def look_at_neighbors(sequence, neighbors, position, wildtype_aa):
        seq_end = len(sequence)
        look_begin, look_end = position-neighbors, min(position+neighbors+1,seq_end)
        wild_pos = sequence[look_begin:look_end].find(wildtype_aa)
        if wild_pos != -1:
            wild_pos = (position-neighbors)+wild_pos

        return wild_pos
    if neighbors>0:
        neighbor_pos = look_at_neighbors(sequence, neighbors, position, wildtype_aa)
        neighbor_pos_chain = look_at_neighbors(sequence, neighbors, position+chain_start, wildtype_aa)
    else:
        neighbor_pos = -1
        neighbor_pos_chain = -1

    if (len(sequence)>position and wildtype_aa == sequence[position]):
        return mutation_code
    elif (len(sequence)>position+chain_start 
        and wildtype_aa == sequence[position+chain_start]):
        return wildtype_aa+str(position+chain_start)+mutation_aa
    elif neighbor_pos != -1:
        return wildtype_aa+str(neighbor_pos)+mutation_aa
    elif neighbor_pos_chain != -1:
        return wildtype_aa+str(neighbor_pos_chain)+mutation_aa
    else:
        return ""


def apply_correct_mutation_code(df: pd.DataFrame):
    def apply_correction(row):
        correct_code = correct_mutation_code(row["mutation_code"], 
                                                    row["sequence"], row["chain_start"], 
                                                    neighbors=1)
        row["mutation_code"] = correct_code
        return row

    df = df.apply(apply_correction, axis=1)
    return df

def coherent_uniprot(mutation_code: str, pdb_ids=[], uniprot_ids=[], 
                        pdb_uniprot_mapping_df=None, uniprot_infos_df=None):
    """
    no need to call this function if there is only one uniprot id & you trust the db it's from
    takes a coherent solution among multiple pdb_ids and/or in respect to the mutation_code
    this function calls get_uniprot_infos to get infos

    @returns 
        - the (pdb_id, uniprot_id) of a coherent solution that match the mutation_code and a pdb_id
        IF all pdb_ids refer to the same uniprot, 
        - this one directly w/ the first pdb_id: (first_pdb_id, uniprot_id)
        IF no coherent pdb_id is found
        - ("", "")
    """
    # get associated infos for uniprot ids and pdb_ids (in case no uniprot id is present):
    uniprot_infos = get_uniprot_infos(pdb_ids=pdb_ids, uniprot_ids=uniprot_ids,
                                        pdb_uniprot_mapping_df=pdb_uniprot_mapping_df, uniprot_infos_df=uniprot_infos_df)

    if pdb_ids:
        # get unique occurences of uniprot_ids
        if uniprot_ids:
            unique_uniprot_ids = list(set(uniprot_ids))
        else:
            unique_uniprot_ids = list(set([v.get("uniprot") for k,v in uniprot_infos.items()]))
        # if there is only one uniprot id, we return this one
        if len(unique_uniprot_ids)==1:
            return (pdb_ids[0], unique_uniprot_ids[0])

        # multiple possible uniprot ids
        # we simply take the first one which is coherent, if uniprot_ids was given we make sure uniprot is in it.
        for pdb, infos in uniprot_infos.items():
            if pdb not in pdb_ids:
                # case when we have uniprot ids, simply skip those
                continue

            corrected_code = correct_mutation_code(mutation_code, infos["sequence"], infos["chain_start"])
            if (corrected_code 
                and (uniprot_ids==[] or infos["uniprot"] in uniprot_ids)
                ):
                # this means the mutation_code (or the corrected one ie. taking the chain_start into account)
                # is coherent => we take this one
                return (pdb, infos["uniprot"])

    elif uniprot_ids:
        # no pdb_ids were given, in that case we just look for the 1st one of the uniprot
        # which has a coherent sequence
        for uniprot, infos in uniprot_infos.items():
            if uniprot not in uniprot_ids:
                # case when we have pdb ids, simply skip those
                continue

            corrected_code = correct_mutation_code(mutation_code, infos["sequence"], infos["chain_start"])
            if corrected_code:
                # this means the mutation_code (or the corrected one ie. taking the chain_start into account)
                # is coherent => we take this one
                return ("", infos["uniprot"])
    
    # no coherent mutation code was found:
    return ("", "")

def add_uniprot_infos(df: pd.DataFrame, 
                        uniprot_infos_cols=['sequence', 'length', 'molWeight', 
                                            'countByFeatureType', 'chain_start', 
                                            'chain_end', 'AlphaFoldDB']):
    # add uniprot infos to a df
    # the df must have a "uniprot" column

    # load the uniprot pdb infos from the csv
    pdb_uniprot_mapping_df = pd.read_csv("./data/main_dataset/pdb_uniprot_mapping.csv")
    uniprot_infos_df = pd.read_csv("./data/main_dataset/uniprot_infos.csv")

    # 1st we add the columns
    for col in uniprot_infos_cols:
        df[col] = np.nan

    # 2nd we add infos via an apply function
    def add_infos(row):
        uniprot_ids = [] 
        pdb_ids = []

        if (type(row["uniprot"])==type("")):
            uniprot_ids = row["uniprot"].split(" ")
        if (type(row["PDB_wild"])==type("")):
            pdb_ids = row["PDB_wild"].split('|') 
        
        (coherent_pdb_id, coherent_uniprot_id) = coherent_uniprot(row["mutation_code"], pdb_ids=pdb_ids, uniprot_ids=uniprot_ids,
                                                                    pdb_uniprot_mapping_df=pdb_uniprot_mapping_df, uniprot_infos_df=uniprot_infos_df)
        
        uniprot_infos = get_uniprot_infos(uniprot_ids=[coherent_uniprot_id], 
                                            pdb_uniprot_mapping_df=pdb_uniprot_mapping_df, uniprot_infos_df=uniprot_infos_df)
        infos = uniprot_infos.get(coherent_uniprot_id)
        for col in uniprot_infos_cols:
            row[col] = infos.get(col) if infos else np.nan
        return row

    df = df.apply(add_infos, axis=1)
    return df

