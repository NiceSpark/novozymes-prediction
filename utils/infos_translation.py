DSSP_codes_Secondary_Structure = {
    "H": {"value": 0, "name": "Alpha helix (4-12)"},
    "B": {"value": 1, "name": "Isolated beta-bridge residue"},
    "E": {"value": 2, "name": "Strand"},
    "G": {"value": 3, "name": "3-10 helix"},
    "I": {"value": 4, "name": "Pi helix"},
    "T": {"value": 5, "name": "Turn"},
    "S": {"value": 6, "name": "Bend"},
    "-": {"value": 7, "name": "None"},
}

DSSP_Data_Keys = [
    "DSSP index",
    "Amino acid",
    "Secondary structure",
    "Relative ASA",
    "Phi",
    "Psi",
    "NH->O_1_relidx",
    "NH->O_1_energy",
    "O->NH_1_relidx",
    "O->NH_1_energy",
    "NH->O_2_relidx",
    "NH->O_2_energy",
    "O->NH_2_relidx",
    "O->NH_2_energy",
]

aa_map = dict(Alanine=("Ala", "A"), Arginine=("Arg", "R"), Asparagine=("Asn", "N"), Aspartic_Acid=("Asp", "D"),
              Cysteine=("Cys", "C"), Glutamic_Acid=("Glu", "E"), Glutamine=("Gln", "Q"), Glycine=("Gly", "G"),
              Histidine=("His", "H"), Isoleucine=("Ile", "I"), Leucine=("Leu", "L"), Lysine=("Lys", "K"),
              Methionine=("Met", "M"), Phenylalanine=("Phe", "F"), Proline=("Pro", "P"), Serine=("Ser", "S"),
              Threonine=("Thr", "T"), Tryptophan=("Trp", "W"), Tyrosine=("Tyr", "Y"), Valine=("Val", "V"),
              Deletion=("del", "-"), Insertion=("ins", "^"))
n_aa = len(aa_map)
aa_chars_ordered = sorted([v[1] for v in aa_map.values()])
aa_long2tri = {k: v[0] for k, v in aa_map.items()}
aa_long2char = {k: v[1] for k, v in aa_map.items()}
aa_tri2long = {v: k for k, v in aa_long2tri.items()}
aa_char2long = {v: k for k, v in aa_long2char.items()}
aa_char2int = {_aa: i for i, _aa in enumerate(aa_chars_ordered)}
aa_int2char = {v: k for k, v in aa_char2int.items()}
