import torch
import esm
import gc
import pandas as pd
import numpy as np
import tqdm
from cuml import PCA
from biopandas.pdb import PandasPdb
from scipy.special import softmax
from scipy.stats import entropy

PCA_CT = 16  # random sample size per protein to fit PCA with
SUBSET_DUPLICATES_NO_PH = ["uniprot", "wild_aa", "mutation_position",
                           "mutated_aa", "sequence"]


def load_esm_model():
    token_map = {'L': 0, 'A': 1, 'G': 2, 'V': 3, 'S': 4, 'E': 5, 'R': 6, 'T': 7, 'I': 8, 'D': 9, 'P': 10,
                 'K': 11, 'Q': 12, 'N': 13, 'F': 14, 'Y': 15, 'M': 16, 'H': 17, 'W': 18, 'C': 19}
    t_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    t_model.eval()  # disables dropout for deterministic results
    print("loaded model")
    return t_model, token_map, batch_converter


def extract_embeddings(df, all_sequences, max_cuda_seq_len,
                       embeddings, t_model, device, batch_converter):
    # EXTRACT TRANSFORMER EMBEDDINGS FOR TRAIN AND TEST WILDTYPES
    print("Extracting embeddings from proteins...")

    all_seq_embed_pool = embeddings["all_seq_embed_pool"]
    all_seq_embed_local = embeddings["all_seq_embed_local"]
    all_seq_embed_by_position = embeddings["all_seq_embed_by_position"]
    all_seq_prob = embeddings["all_seq_prob"]

    sequences_too_big_for_cuda = []

    for i, seq in tqdm.tqdm(enumerate(all_sequences)):
        # EXTRACT EMBEDDINGS, MUTATION PROBABILITIES, ENTROPY

        # check the device is coherent with protein length
        if (str(device) == "cuda" and len(seq) > max_cuda_seq_len):
            # if the protein is too big, don't try (we will do it with a cpu later)
            sequences_too_big_for_cuda.append(seq)
            continue
        elif (str(device) == "cpu" and len(seq) <= max_cuda_seq_len):
            continue

        data = [("_", seq)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = t_model(batch_tokens, repr_layers=[33])
        # go from 33 to 20 (1 per amino acid)
        logits = (results["logits"].detach().cpu().numpy()[0, ].T)[4:24, 1:-1]
        all_seq_prob[i] = softmax(logits, axis=0)
        results = results["representations"][33].detach().cpu().numpy()

        # SAVE EMBEDDINGS
        all_seq_embed_local[i] = results
        all_seq_embed_pool[i, ] = np.mean(results[0, :, :], axis=0)

        # TEMPORARILY SAVE LOCAL MUTATION EMBEDDINGS
        mutation_positions = df.loc[df.sequence == seq,
                                    "mutation_position"].unique().astype(int)

        # the goal here is to fit the pca on the concat of all embeddings,
        # therefore if one protein has 1000 single mutation it will appear 1000 times
        # and we will overfit the pca to this protein
        # => we choose max PCA_CT single mutations
        if len(mutation_positions) > PCA_CT:
            mutation_positions = np.random.choice(
                mutation_positions, PCA_CT, replace=False)
        for j in mutation_positions:
            all_seq_embed_by_position[i] = results[0, j+1, :]

        del batch_tokens, results
        gc.collect()
        torch.cuda.empty_cache()

    embeddings = {
        "all_seq_embed_pool": all_seq_embed_pool,
        "all_seq_embed_local": all_seq_embed_local,
        "all_seq_embed_by_position": all_seq_embed_by_position,
        "all_seq_prob": all_seq_prob,
    }

    return embeddings, sequences_too_big_for_cuda


def compute_embeddings_pca(df, t_model, batch_converter, max_cuda_seq_len):
    all_sequences = df.sequence.unique()
    embeddings = {
        "all_seq_embed_pool": np.zeros((len(all_sequences), 1280)),
        "all_seq_embed_local": [None]*len(all_sequences),
        "all_seq_embed_by_position": [None]*len(all_sequences),
        "all_seq_prob": [None]*len(all_sequences),
    }

    # first we do 'small' proteins with cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_model.to(device)
    print(device)
    embeddings, sequences_too_big_for_cuda = extract_embeddings(df, all_sequences,
                                                                max_cuda_seq_len, embeddings,
                                                                t_model, device, batch_converter)

    # then we do the biggest proteins with cpu, if needed
    if len(sequences_too_big_for_cuda) > 0:
        device = torch.device("cpu")
        t_model.to(device)
        print(device)
        print(len(sequences_too_big_for_cuda))
        embeddings, sequences_too_big_for_cuda = extract_embeddings(df, all_sequences,
                                                                    max_cuda_seq_len, embeddings,
                                                                    t_model, device, batch_converter)

    # RAPIDS PCA
    # The transformer embeddings have dimension 1280.
    # Since we only have a few thousand rows of train data,
    # that is too many features to include all of them in our XGB model.
    # Furthermore, we want to use local, pooling, and delta embeddings.
    # Which would be 3x1280. To prevent our model from overfitting
    # as a result of the "curse of dimensionality",
    # we reduce the dimension of embeddings using RAPIDS PCA.

    # set sequence_to_embed_mapping
    sequence_to_embed_mapping = {seq: i for i, seq in enumerate(all_sequences)}
    # create stack
    all_seq_embed_by_position = np.stack(
        embeddings.pop("all_seq_embed_by_position"))
    pca_pool = PCA(n_components=32)
    pca_embeds = pca_pool.fit_transform(
        embeddings.pop("all_seq_embed_pool").astype("float32"))
    pca_local = PCA(n_components=16)
    pca_local.fit(all_seq_embed_by_position.astype("float32"))

    # we delete all_seq_embed_by_position: we only used it to fit the pca_local
    del all_seq_embed_by_position
    _ = gc.collect()

    return embeddings, sequence_to_embed_mapping, pca_pool, pca_embeds, pca_local, sequences_too_big_for_cuda


def compute_embeddings_only(df, t_model, batch_converter, pca_pool, max_cuda_seq_len):
    all_sequences = df.sequence.unique()
    embeddings = {
        "all_seq_embed_pool": np.zeros((len(all_sequences), 1280)),
        "all_seq_embed_local": [None]*len(all_sequences),
        "all_seq_embed_by_position": [None]*len(all_sequences),
        "all_seq_prob": [None]*len(all_sequences),
    }

    # first we do 'small' proteins with cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_model.to(device)
    print(device)
    embeddings, _ = extract_embeddings(df, all_sequences,
                                       max_cuda_seq_len, embeddings,
                                       t_model, device, batch_converter)

    # set sequence_to_embed_mapping
    sequence_to_embed_mapping = {seq: i for i, seq in enumerate(all_sequences)}

    # create stack
    pca_embeds = pca_pool.transform(
        embeddings.pop("all_seq_embed_pool").astype("float32"))

    _ = gc.collect()

    return embeddings, sequence_to_embed_mapping, pca_embeds


def add_embbeddings(row, sequence_to_embed_mapping, embeddings,
                    pca_local, pca_pool, pca_embeds, token_map,
                    t_model, device, batch_converter, max_cuda_seq_len,
                    errors):
    try:
        ##################
        # ROW - IS ROW FROM DOWNLOADED TRAIN CSV
        ##################
        # pdb_map = {x: y for x, y in zip(all_pdb, range(len(all_pdb)))}
        atom_df = PandasPdb().read_pdb(row.alphafold_path)
        atom_df = atom_df.df['ATOM']

        residue_atoms = atom_df.loc[(
            atom_df.residue_number == row.mutation_position)].reset_index(drop=True)

        # FEATURE ENGINEER
        if len(residue_atoms) > 0:

            # check the device is coherent with protein length
            if (str(device) == "cuda" and len(row.sequence) > max_cuda_seq_len):
                # if the protein is too big, don't try (we will do it with a cpu later)
                return row
            elif (str(device) == "cpu" and len(row.sequence) <= max_cuda_seq_len):
                return row

            # GET MUTANT EMBEDDINGS
            mutated_sequence = (row.sequence[:row.mutation_position] +
                                row.mutated_aa+row.sequence[row.mutation_position+1:])
            data = [("_", mutated_sequence)]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)
            with torch.no_grad():
                results = t_model(batch_tokens, repr_layers=[33])
            results = results["representations"][33].cpu().numpy()
            mutant_local = pca_local.transform(
                results[:1, row.mutation_position+1, :])[0, ]
            mutant_pool = np.mean(results[:1, :, :], axis=1)
            mutant_pool = pca_pool.transform(mutant_pool)[0, ]

            # TRANSFORMER ESM EMBEDDINGS
            wild_local = pca_local.transform(
                embeddings["all_seq_embed_local"][sequence_to_embed_mapping[row.sequence]][:1, row.mutation_position+1, :])[0, ]
            wild_pool = pca_embeds[sequence_to_embed_mapping[row.sequence], ]
            for k in range(32):
                row[f"esm_pca_pool_{k}"] = mutant_pool[k] - wild_pool[k]
                if k >= 16:
                    continue
                row[f"esm_pca_wild_{k}"] = wild_local[k]
                row[f"esm_pca_mutant_{k}"] = mutant_local[k]
                row[f"esm_pca_local_{k}"] = mutant_local[k] - wild_local[k]

            # TRANSFORMER MUTATION PROBS AND ENTROPY
            row["esm_mutation_probability"] = embeddings["all_seq_prob"][sequence_to_embed_mapping[row.sequence]
                                                                         ][token_map[row.mutated_aa], row.mutation_position+1]
            row["esm_mutation_entropy"] = entropy(
                embeddings["all_seq_prob"][sequence_to_embed_mapping[row.sequence]][:, row.mutation_position+1])

            del batch_tokens, results, mutant_local, mutant_pool, wild_local, wild_pool
            gc.collect()
            torch.cuda.empty_cache()
    except Exception as e:
        errors.append(
            f"error occured for {row.uniprot} {row.mutation_position} {row.mutated_aa}")

    return row


def add_embeddings_to_df(df, sequences_too_big_for_cuda, sequence_to_embed_mapping,
                         embeddings, t_model, pca_local, pca_pool, pca_embeds,
                         batch_converter, token_map, max_cuda_seq_len):
    df.mutation_position = df.mutation_position.astype(int)
    errors = []
    # first we do 'small' proteins with cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_model.to(device)
    print(device)
    df = df.apply(lambda row: add_embbeddings(
        row, sequence_to_embed_mapping, embeddings,
        pca_local, pca_pool, pca_embeds, token_map,
        t_model, device, batch_converter, max_cuda_seq_len, errors), axis=1)

    if len(sequences_too_big_for_cuda) > 0:
        # then we do the biggest proteins with cpu if needed
        device = torch.device("cpu")
        t_model.to(device)
        print(device)
        print(len(sequences_too_big_for_cuda))
        df = df.apply(lambda row: add_embbeddings(
            row, sequence_to_embed_mapping, embeddings,
            pca_local, pca_pool, pca_embeds, token_map,
            t_model, device, batch_converter, max_cuda_seq_len, errors), axis=1)

    return df


def update_main_df(row, embeddings_df, new_columns):
    # SUBSET_DUPLICATES_NO_PH = ["uniprot", "wild_aa", "mutation_position",
    #                            "mutated_aa", "sequence"]
    associated_embeddings_row = embeddings_df.loc[
        (embeddings_df["uniprot"] == row["uniprot"]) &
        (embeddings_df["wild_aa"] == row["wild_aa"]) &
        (embeddings_df["mutation_position"] == row["mutation_position"]) &
        (embeddings_df["mutated_aa"] == row["mutated_aa"]) &
        (embeddings_df["sequence"] == row["sequence"])
    ]
    if len(associated_embeddings_row) == 1:
        associated_embeddings_row = associated_embeddings_row.iloc[0, :]
        for col in new_columns:
            row[col] = associated_embeddings_row[col]

    return row


def add_columns(df, new_columns):
    new_columns_df = pd.DataFrame(columns=new_columns)
    df = pd.concat([df, new_columns_df], axis=1)
    return df


def add_esm_features(main_df, only_ddg=True, max_cuda_seq_len=7000, save_embeddings_df=True, use_saved_embeddings=False):
    # add new columns
    new_columns = [f"esm_pca_pool_{k}" for k in range(32)]
    new_columns += [f"esm_pca_wild_{k}" for k in range(16)]
    new_columns += [f"esm_pca_mutant_{k}" for k in range(16)]
    new_columns += [f"esm_pca_local_{k}" for k in range(16)]
    new_columns += ["esm_mutation_probability", "esm_mutation_entropy"]

    add_columns(main_df, new_columns)

    if use_saved_embeddings:
        df = pd.read_csv("embeddings.csv")
    else:
        t_model, token_map, batch_converter = load_esm_model()
        df = main_df.copy()
        df.drop_duplicates(subset=SUBSET_DUPLICATES_NO_PH, inplace=True)

        if only_ddg:
            df = df[~(df.ddG.isna())]
        add_columns(df, new_columns)

        embeddings, seq2embeds, pca_pool, pca_embeds, pca_local, seq2big = compute_embeddings_pca(
            df, t_model, batch_converter, max_cuda_seq_len)

        df = add_embeddings_to_df(df, seq2big, seq2embeds, embeddings,
                                  t_model, pca_local, pca_pool, pca_embeds,
                                  batch_converter, token_map, max_cuda_seq_len)

    # delete all rows with nan esm features
    for col in new_columns:
        df = df[~(df[col].isna())]

    if save_embeddings_df:
        df.to_csv("embeddings.csv", index=False)

    main_df = main_df.apply(lambda row: update_main_df(row, df, new_columns),
                            axis=1)
    return main_df


def submission_compute_pca(main_df, submission_df, only_ddg=True, max_cuda_seq_len=7000):

    t_model, token_map, batch_converter = load_esm_model()
    df = main_df.copy()
    df.drop_duplicates(subset=SUBSET_DUPLICATES_NO_PH, inplace=True)

    if only_ddg:
        df = df[~(df.ddG.isna())]

    # we need to have the same pca as the training dataset
    _, _, pca_pool, pca_embeds, pca_local, seq2big = compute_embeddings_pca(
        df, t_model, batch_converter, max_cuda_seq_len)

    # we need the embeddings for the test dataset
    embeddings, seq2embeds, pca_embeds = compute_embeddings_only(
        submission_df, t_model, batch_converter, pca_pool, max_cuda_seq_len)

    context = {
        "seq2big": seq2big,
        "seq2embeds": seq2embeds,
        "embeddings": embeddings,
        "t_model": t_model,
        "pca_local": pca_local,
        "pca_pool": pca_pool,
        "pca_embeds": pca_embeds,
        "batch_converter": batch_converter,
        "token_map": token_map,
        "max_cuda_seq_len": max_cuda_seq_len,
    }
    return context


def submission_add_esm_features(submission_df, context):
    seq2big = context["seq2big"]
    seq2embeds = context["seq2embeds"]
    embeddings = context["embeddings"]
    t_model = context["t_model"]
    pca_local = context["pca_local"]
    pca_pool = context["pca_pool"]
    pca_embeds = context["pca_embeds"]
    batch_converter = context["batch_converter"]
    token_map = context["token_map"]
    max_cuda_seq_len = context["max_cuda_seq_len"]

    new_columns = [f"esm_pca_pool_{k}" for k in range(32)]
    new_columns += [f"esm_pca_wild_{k}" for k in range(16)]
    new_columns += [f"esm_pca_mutant_{k}" for k in range(16)]
    new_columns += [f"esm_pca_local_{k}" for k in range(16)]
    new_columns += ["esm_mutation_probability", "esm_mutation_entropy"]
    submission_df = add_columns(submission_df, new_columns)

    submission_df = add_embeddings_to_df(submission_df, seq2big, seq2embeds, embeddings,
                                         t_model, pca_local, pca_pool, pca_embeds,
                                         batch_converter, token_map, max_cuda_seq_len)
    return submission_df
