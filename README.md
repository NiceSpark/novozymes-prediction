# Novozymes enzymes stability predictions

[Kaggle competition overview page](https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/overview)

## Dataset Description

In this competition, you are asked to develop models that can predict the ranking of protein thermostability (as measured by melting point, tm) after single-point amino acid mutation and deletion.

For the training set, the protein thermostability (experimental melting temperature) data includes natural sequences, as well as engineered sequences with single or multiple mutations upon the natural sequences. The data are mainly from different sources of published studies such as Meltome atlas—thermal proteome stability across the tree of life. Many other public datasets exist for protein stability; please see the competition Rule 7C for external data usage requirements. There are also other publicly available methods which can predict protein stabilities such as ESM, EVE and Rosetta etc., without using the provided training set. These methods may also be used as part of the competition.

The test set contains experimental melting temperature of over 2,413 single-mutation variant of an enzyme (GenBank: KOC15878.1), obtained by Novozymes A/S. The amino acid sequence of the wild type is:

VPVNPEPDATSVENVALKTGSGDSQSDPIKADLEVKGQSALPFDVDCWAILCKGAPNVLQRVNEKTKNSNRDRSGANKGPFKDPQKWGIKALPPKNPSWSAQDFKSPEEYAFASSLQGGTNAILAPVNLASQNSQGGVLNGFYSANKVAQFDPSKPQQTKGTWFQITKFTGAAGPYCKALGSNDKSVCDKNKNIAGDWGFDPAKWAYQYDEKNNKFNYVGK
Files

**train.csv** - the training data, with columns as follows:
seq_id: unique identifier of each protein variants
protein_sequence: amino acid sequence of each protein variant. The stability (as measured by tm) of protein is determined by its protein sequence. (Please note that most of the sequences in the test data have the same length of 221 amino acids, but some of them have 220 because of amino acid deletion.)
pH: the scale used to specify the acidity of an aqueous solution under which the stability of protein was measured. Stability of the same protein can change at different pH levels.
data_source: source where the data was published
tm: target column. Since only the spearman correlation will be used for the evaluation, the correct prediction of the relative order is more important than the absolute tm values. (Higher tm means the protein variant is more stable.)

**train_updates_20220929.csv** - corrected rows in train, please see this forum post for details

**test.csv** - the test data; your task is to predict the target tm for each protein_sequence (indicated by a unique seq_id)

**sample_submission.csv** - a sample submission file in the correct format, with seq_id values corresponding to test.csv

**wildtype_structure_prediction_af2.pdb** - the 3 dimensional structure of the enzyme listed above, as predicted by AlphaFold

## Resources:

See the Notion export in resources/ for more infos

Some additionnal prediction of protein folding were done using [alphafold notebook](https://colab.research.google.com/github/deepmind/alphafold/blob/main/notebooks/AlphaFold.ipynb#scrollTo=woIxeCPygt7K)

## Dependencies

Some features are computed using msms, see [this link for download](https://ccsb.scripps.edu/msms/downloads/), as well as [mgltools from the same authors](https://ccsb.scripps.edu/mgltools/downloads/)

Some features are computed using DeMaSK, see their [README](https://github.com/Singh-Lab/DeMaSk)
DeMaSK itself is dependent on [Blastp](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/) and [UniRef90 in fasta format](https://www.uniprot.org/help/downloads)
