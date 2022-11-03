import pandas as pd
import numpy as np
from utils.file_utils import open_json, write_json


def update_linked_uniprot(mapped_uniprots, linked_uniprot_mapping):
    for uniprot in mapped_uniprots:
        linked_uniprots = linked_uniprot_mapping.get(uniprot, [])
        linked_uniprots += mapped_uniprots
        linked_uniprots = list(set(linked_uniprots))  # only unique values
        linked_uniprots.pop(linked_uniprots.index(
            uniprot))  # no need for key in value
        linked_uniprot_mapping[uniprot] = linked_uniprots  # update mapping
    return linked_uniprot_mapping


def update_pdb_uniprot_mapping(uniprot_infos_path,
                               pdb_uniprot_mapping_path,
                               linked_uniprot_mapping_path):
    local_uniprot_infos = open_json(uniprot_infos_path)
    pdb_uniprot_mapping = open_json(pdb_uniprot_mapping_path)
    linked_uniprot_mapping = open_json(linked_uniprot_mapping_path)
    l = len(pdb_uniprot_mapping)

    for uniprot in local_uniprot_infos:
        pdbs = local_uniprot_infos[uniprot].get("pdbs", "")

        if pdbs == "":
            continue

        sep = '|' if '|' in pdbs else ' '
        for pdb in pdbs.split(sep):
            mapped_uniprots = pdb_uniprot_mapping.get(pdb, [])
            if uniprot not in mapped_uniprots:
                mapped_uniprots.append(uniprot)
                pdb_uniprot_mapping[pdb] = mapped_uniprots
                # if needed add mapping to linked uniprot,
                # ie 2 uniprot mapped to same pdb
                if len(mapped_uniprots) > 1:
                    linked_uniprot_mapping = update_linked_uniprot(
                        mapped_uniprots, linked_uniprot_mapping)

    print(f"added {len(pdb_uniprot_mapping)-l} entries to pdb_uniprot_mapping")
    write_json(pdb_uniprot_mapping_path, pdb_uniprot_mapping)
    write_json(linked_uniprot_mapping_path, linked_uniprot_mapping)


def update_sequence_uniprot_mapping(uniprot_infos_path,
                                    sequence_uniprot_mapping_path,
                                    linked_uniprot_mapping_path):
    local_uniprot_infos = open_json(uniprot_infos_path)
    sequence_uniprot_mapping = open_json(sequence_uniprot_mapping_path)
    linked_uniprot_mapping = open_json(linked_uniprot_mapping_path)

    l = len(sequence_uniprot_mapping)

    for uniprot in local_uniprot_infos:
        sequence = local_uniprot_infos[uniprot].get("sequence", "")

        if sequence == "":
            continue

        mapped_uniprots = sequence_uniprot_mapping.get(sequence, [])
        if uniprot not in mapped_uniprots:
            mapped_uniprots.append(uniprot)
            sequence_uniprot_mapping[sequence] = mapped_uniprots
            # if needed add mapping to linked uniprot,
            # ie 2 uniprot mapped to same pdb
            if len(mapped_uniprots) > 1:
                linked_uniprot_mapping = update_linked_uniprot(
                    mapped_uniprots, linked_uniprot_mapping)

    print(
        f"added {len(sequence_uniprot_mapping)-l} entries to sequence_uniprot_mapping")
    write_json(sequence_uniprot_mapping_path, sequence_uniprot_mapping)
    write_json(linked_uniprot_mapping_path, linked_uniprot_mapping)
