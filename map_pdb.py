import tqdm
from utils.file_utils import open_json, write_json

pdb_no_uniprot = open_json(
    "./data/main_dataset_creation/mapping/pdb_no_uniprot.json")
pdb_uniprot_mapping2 = open_json(
    "./data/main_dataset_creation/mapping/pdb_uniprot_mapping_searched_in_db.json")


def add_pdb_to_mapping(uniprot: str, pdb: str, pdb_uniprot_mapping: dict):
    mapped_uniprots = pdb_uniprot_mapping.get(pdb, [])
    if uniprot not in mapped_uniprots:
        mapped_uniprots.append(uniprot)
        pdb_uniprot_mapping[pdb] = mapped_uniprots
    return pdb_uniprot_mapping


with open("/media/tom/ML_WORK/uniprot_id_mapping/idmapping.dat") as fp:
    for line in tqdm.tqdm(fp):
        if ("PDB" in line):
            values = line.split('\t')
            uniprot = values[0]
            pdb = values[2].strip()
            if pdb in pdb_no_uniprot:
                pdb_uniprot_mapping2 = add_pdb_to_mapping(
                    uniprot, pdb, pdb_uniprot_mapping2)

write_json(
    "./data/main_dataset_creation/mapping/pdb_uniprot_mapping_searched_in_db.json", pdb_uniprot_mapping2)
