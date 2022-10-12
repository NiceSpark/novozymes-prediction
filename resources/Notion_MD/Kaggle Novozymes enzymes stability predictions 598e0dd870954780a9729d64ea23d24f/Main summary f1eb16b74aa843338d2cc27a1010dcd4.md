# Main summary

### Features

**Step of redundant features removal needs to be done !**

![List of AI-based DDG predictors studied.png](Resources%20summary%208e1ddd141d1d433a9f8786d5b0eb8f55/List_of_AI-based_DDG_predictors_studied.png)

- raw list:
    - amino acid sequences
    - protein chain length
    - molecular weight
    - physicochemical features
        - burial and tight packing of hydrophobic residues
        - hydrogen bonds & other electrostatic interactions
        - conformational entropy
        - bond strain
        - surface electrostatics
        - isoelectric point
        - CHNSO (carbon, hydrogen, nitrogen, sulphur, oxygen) counts for elem types & frequencies
        - 6 amino acid group counts and freq
        - count and freq of -/+ charged hydrophilic & hydrophobic residues
        - dipeptide counts
    - ejection of ordered solvent
    - hydropathy
    - structural, and composition features that describe properties of entire proteins
        - flexibility
    - learned (via self-supervised training) likelihood distribution in amino sequence
- properties:
    - symmetry $∆∆GMW = −∆∆GMW$
- features type used:
    - residue (mutated aa) specifics:
        - ΔFlex (change in flexibility)
        - MutI (mutability index of the native residue)
        - ΔMW (change in molecular weight)
        - Positive, Negative Hyd (residue hydrophobicity)
        - ΔHyd (change in Hyd)
        - ΔIP
        - Residue environment
        - Residue type
        - Vol (residue volume)
        - ΔVol (change in residue volume upon mutation)
        - Residue distances/orientations
        - Neighbors (type of residues in the neighborhood along the sequence)
    - ASA (solvent accessible surface area)
    - RSA (Relative ASA)
    - Statistical potentials
    - Atomic distance patterns
    - Graph based signatures
    - PSize (protein size)
    - SS (secondary structure)
    - 4-Body statistical potential
    - depth (surface, under surface, buried)
    - contact potential
    - EvolInfo (Evolutionary information from protein families)
    - BL62 (BLOSUM62 scoring matrix for aa substitution)
    - Environment specific substitution freq
    - Energy functions
    - Homology modeling
    - H-bonds
    - Aromatic
    - H-bond donor/acceptor
    - Charged
    - others (a bit weird)
        - Leu (Leucine, one of the 20 aa)
- Some features computed using:
    - PROFEAT (structural & physicochemical)
    - PROTEIN RECON (protein charge density based electronic porperties)
    - ProtDCal (generate sequence based descriptors)
- NLP inspiration ⇒ feature engineering ideas:
    - Count Vectorizer on character based scale
    - TFIDF Vectorizer on a character based scale
    - n-gram based analysis of the amino acids
    - some other custom amino acid pattern search
    - addition of pH feature

![Untitled](Main%20summary%20f1eb16b74aa843338d2cc27a1010dcd4/Untitled.png)

---

### Already used Datasets

- **ThermoMutDB:**
    - includes ProTherm cleaned and pruned [(link)](http://structure.bmc.lu.se/VariBench/stability.php)
        - (S2648, Q3421  and Varibench)
- **FireProtDB**
- Ssym ⇒ and Ssym+ (Ssym + ThermoMut entries that works for Ssym)
- PON-tstab, and curated PON-tstab
- S669: curated difference btw ThermoMut and (S2648 + Varibench) ⇒ for testing
    - + there exists a generated reverse variants for S669 (to assess the antisymmetry, some structure computed with Rosetta) in a ⇒ dataset of 1338 protein variants.
- pre-trained ACDC-NN-Seq using the predictions of another method, DDGun3D

**About training/test datasets:**

![competition_data_explained.png](Main%20summary%20f1eb16b74aa843338d2cc27a1010dcd4/competition_data_explained.png)

---

### ML implementations

in the literature people usually computes ΔΔG, link between ΔΔG and ΔTm here: [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4796876/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4796876/)

limit of 1 kcal/mol RMSE ⇒ might be hard to go lower (RMSE stagnated in the last 15 years), but **remember the goal is to classify the different mutation between each other, more than computing the correct tm**

- could simply do 2 inputs (both prot seq) ⇒which one is more stable? then merge sort or whatever