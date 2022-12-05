import json
import pandas as pd
import numpy as np
import urllib.request
from .file_utils import open_json, write_json


def add_missing_column(df: pd.DataFrame, columns: list):
    for name in columns:
        if name not in df.columns.to_list():
            df[name] = np.nan

    return df


def save_df(df: pd.DataFrame, name: str):
    df.to_csv(f"./data/main_dataset_creation/{name}.csv", index=False)


def split_mutation_code(mutation_code: str):
    if type(mutation_code) != type("") or len(mutation_code) < 2:
        return None, None, None

    mutation_code = mutation_code.strip()
    if ((' ' in mutation_code) or (',' in mutation_code) or ('|' in mutation_code)):
        # multiple mutations !
        return None, None, None

    wild_aa = mutation_code[0]
    try:
        mutation_position = int(mutation_code[1:-1])
    except:
        return None, None, None

    mutated_aa = mutation_code[-1]

    return wild_aa, mutation_position, mutated_aa


def apply_split_mutation_code(row):
    wild_aa, mutation_position, mutated_aa = split_mutation_code(
        row["mutation_code"])
    row["wild_aa"] = wild_aa
    row["mutation_position"] = mutation_position
    row["mutated_aa"] = mutated_aa
    return row


def correct_mutation_position(wild_aa: str, mutation_position: int, sequence: str,
                              chain_start: int, position_offset_max: int):
    """
    This function check if the mutation_code is coherent

    @returns:
        the correct mutation_code
        or None if no mutation_code is false and cannot be corrected
    """
    try:
        mutation_position = int(mutation_position)
        position_offset_max = int(position_offset_max)
        chain_start = int(chain_start)
    except:
        error = {
            "error": "positions are not all int",
            "mutation_position": mutation_position,
            "position_offset": position_offset,
            "chain_start": chain_start,
            "sequence": sequence,
        }
        print(error)
        errors = open_json("correct_mutation_position_errors.json")
        errors.append(error)
        write_json(errors)
        return None

    if sequence == "":
        return None
    elif mutation_position >= len(sequence):
        # mutation position not coherent with sequence
        return None
    if sequence[mutation_position] == wild_aa:
        return mutation_position
    if ((chain_start > 0) and (len(sequence) > mutation_position+chain_start)
            and (sequence[mutation_position+chain_start] == wild_aa)):
        # add chain start
        return mutation_position+chain_start

    position_offsets = []
    for k in range(1, position_offset_max+1):
        position_offsets.append(k)
        position_offsets.append(-k)
    for position_offset in position_offsets:
        if ((len(sequence) > mutation_position+position_offset) and
                (sequence[mutation_position+position_offset] == wild_aa)):
            return mutation_position+position_offset

        if ((chain_start > 0) and (len(sequence) > mutation_position+chain_start+position_offset) and
                (sequence[mutation_position+chain_start+position_offset] == wild_aa)):
            return mutation_position+chain_start+position_offset
    return None


def get_uniprot_infos_online(uniprot: str):
    """
    get uniprot infos from the web via a simple http request to their restAPI
    if the entry is wrong, or the data in uniprotDB is missing infos we return {}
    """
    if type(uniprot) != type(""):
        return {}
    elif uniprot.upper() == "NAN":
        return {}

    try:
        with urllib.request.urlopen(f"https://rest.uniprot.org/uniprotkb/{uniprot}.json") as url:
            data = json.load(url)
    except Exception as e:
        print(f"Exception: {e}")
        print(f"uniprot: {uniprot}, type: {type(uniprot)}")
        return {
            "pdbs": "",
            "sequence": "",
            "length": 0,
            "chain_start": 0,
            "chain_end": 0,
            "AlphaFoldDB": "",
        }

    # some uniprot entries exists, but without the needed data
    if "sequence" not in data:
        return {
            "pdbs": "",
            "sequence": "",
            "length": 0,
            "chain_start": 0,
            "chain_end": 0,
            "AlphaFoldDB": "",
        }

    sequence = data.get("sequence", {}).get("value")
    features = data.get("features", [])
    chain_location = next(
        (x for x in features if x["type"] == "Chain"), {}).get("location", {})

    databases = data.get("uniProtKBCrossReferences", [])
    pdb_ids = " ".join([x["id"]
                        for x in databases if (x["database"] == "PDB")])

    return {
        "pdbs": pdb_ids,
        "sequence": sequence,
        "length": len(sequence),
        "chain_start": chain_location.get("start", {}).get("value", 0),
        "chain_end": chain_location.get("end", {}).get("value", 0),
        "AlphaFoldDB": " ".join([x["id"] for x in databases if (x["database"] == "AlphaFoldDB")]),
    }


def valid_uniprot(uniprot: str, local_uniprot_infos: dict, wild_aa: str,
                  mutation_position: int, individual_dataset_config: dict, errors: dict):
    """
    get uniprot data from either local storage or from web
    then check whether the info is coherent with the mutation 
    > sequence[position] = wild_aa
    applies offset on position according to the individual dataset config
    returns:
    - data: infos abt the requested uniprot (can be {} if none was found)
    - updated local_uniprot_infos
    - updated mutation_position (can be np.nan if none was found/incoherent)
    - 
    """
    data = local_uniprot_infos.get(uniprot, {})

    if data == {}:
        errors["not_in_local"] += 1
        data = get_uniprot_infos_online(uniprot)
        local_uniprot_infos[uniprot] = data
        # validate the data
        if "sequence" not in data or data["sequence"] != "":
            errors["no_sequence_in_data"] += 1
            return {}, local_uniprot_infos, None, errors

        # index start at 0 => chain start & end -1
        # this means that if one of those is -1 it means there were no data in uniprot DB
        data["chain_start"] -= 1
        data["chain_end"] -= 1

    corrected_mutation_position = correct_mutation_position(wild_aa, mutation_position,
                                                            data.get(
                                                                "sequence", ""),
                                                            data.get(
                                                                "chain_start", 0) if individual_dataset_config["positions"]["add_chain_start"] else 0,
                                                            individual_dataset_config["positions"]["position_offset"])
    if corrected_mutation_position is None:
        errors["wrong_position"] += 1
        return {}, local_uniprot_infos, None, errors

    return data, local_uniprot_infos, corrected_mutation_position, errors


def apply_valid_uniprot(row, local_uniprot_infos: dict, dataset_config: dict, errors: dict):
    """helper function to apply valid_uniprot on a df"""
    if type(row["uniprot"]) != type(""):
        return row

    # we get the valid infos
    data, local_uniprot_infos, mutation_position, errors = valid_uniprot(row["uniprot"], local_uniprot_infos,
                                                                         row["wild_aa"], row["mutation_position"],
                                                                         dataset_config[row["dataset_source"]],
                                                                         errors)

    # we update with the updated position (taking chain start and offset into account)
    if mutation_position is not None:
        row["mutation_position"] = mutation_position

    if data != {}:
        row["infos_found"] = 1

    # we add each data values in the row
    for k, v in data.items():
        row[k] = v

    return row


def apply_infos_from_pdb(row, local_uniprot_infos: dict, pdb_uniprot_mapping: dict,
                         linked_uniprot_mapping: dict, dataset_config: dict,
                         pdb_without_uniprot: dict, errors: dict):
    """helper function to apply infos_from_pdb on a df"""
    # TODO: used linked uniprot mapping

    if row["infos_found"]:
        return row

    pdbs = row["pdbs"]
    data = {}
    mutation_position = None

    if type(pdbs) != type("") or pdbs == "":
        errors["no_pdb"] += 1
        return row

    # we look for uniprots linked to our pdb:
    sep = '|' if '|' in pdbs else ' '
    for pdb in pdbs.split(sep):
        if data != {}:
            # as soon as we found at least one coherent data we exit the loop
            break

        mapped_uniprots = pdb_uniprot_mapping.get(pdb.upper(), [])
        for uniprot in mapped_uniprots:
            # we get the first valid infos
            # TODO: maybe we can do better than just taking the first one ?
            data, local_uniprot_infos, mutation_position, errors = valid_uniprot(uniprot, local_uniprot_infos,
                                                                                 row["wild_aa"], row["mutation_position"],
                                                                                 dataset_config[row["dataset_source"]],
                                                                                 errors)
            if data != {}:
                # as soon as we found at least one coherent data we exit the loop
                break

    # we update with the updated position (taking chain start and offset into account)
    if mutation_position is not None:
        row["mutation_position"] = mutation_position

    if (data != {} and mutation_position is not None):
        row["uniprot"] = uniprot
        row["infos_found"] = 1
    else:
        # print(f"no uniprot found for: {pdbs.split(sep)}")
        for pdb in pdbs.split(sep):
            pdb = pdb.upper()
            if (pdb not in pdb_uniprot_mapping) and (pdb not in pdb_without_uniprot):
                pdb_without_uniprot.append(pdb)

    # we add each data values in the row
    for k, v in data.items():
        row[k] = v

    return row


def apply_infos_from_sequence(row, local_uniprot_infos: dict, sequence_uniprot_mapping: dict,
                              linked_uniprot_mapping: dict, dataset_config: dict,
                              sequence_without_uniprot: dict, errors: dict):
    """helper function to apply infos_from_pdb on a df"""
    # TODO: used linked uniprot mapping

    if row["infos_found"]:
        return row

    sequence = row["sequence"]
    data = {}
    mutation_position = None

    if type(sequence) != type("") or sequence == "":
        errors["no_sequence"] += 1
        return row

    # we look for uniprots linked to our sequence:
    mapped_uniprots = sequence_uniprot_mapping.get(sequence, [])
    for uniprot in mapped_uniprots:
        # we get the first valid infos
        # TODO: maybe we can do better than just taking the first one ?
        data, local_uniprot_infos, mutation_position, errors = valid_uniprot(uniprot, local_uniprot_infos,
                                                                             row["wild_aa"], row["mutation_position"],
                                                                             dataset_config[row["dataset_source"]],
                                                                             errors)
        if data != {}:
            # as soon as we found at least one coherent data we exit the loop
            break

    # we update with the updated position (taking chain start and offset into account)
    if mutation_position is not None:
        row["mutation_position"] = mutation_position

    if (data != {} and mutation_position is not None):
        row["uniprot"] = uniprot
        row["infos_found"] = 1
    else:
        # print(f"no uniprot found for: {sequence}")
        if sequence not in sequence_without_uniprot:
            sequence_without_uniprot.append(sequence)

    # we add each data values in the row
    for k, v in data.items():
        row[k] = v

    return row
