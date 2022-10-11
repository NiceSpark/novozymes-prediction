# Resources summary

# Articles

---

### Protein stability: computation, sequence statistics, and new experimental methods (2015)

even a single mutation can significantly destabilize or unfold a protein.

- key forces that underlie protein stability are fairly well understood
    - burial and tight packing of hydrophobic residues
    - ejection of ordered solvent
    - hydrogen bonds & other electrostatic interactions
    - conformational entropy
    - bond strain
    - burial of charged residues
    - surface electrostatics
    
- good at calculating geometric parameters, but electrostatics calculations are greatly hampered by how to treat solvation
    
    proteins respond to mutations more by subtle movements of the backbone than adopting unfavorable
    side chain rotamers
    

No single reason emerges for the high stability of variants, very high protein stability is more a function of getting many subtle factors right than simply maximizing a single factor (although core packing remains the most important single effect)

- programs written or used specifically to estimate the ΔΔG: (but ΔΔG errors and sign incorrect about 25% of the time)
    - Rosetta
    - FoldX
    - Eris
    - CC/PBSA

One successful method to generate stable proteins is SCHEMA, which seeks to break proteins down into fragments, the structure of which will not be perturbed upon recombination into another protein

---

### ProTstab – predictor for cellular protein stability (2019)

- methods have been developed for the prediction of protein stability, especially Tm [(see this article for more)](Resources%20List%20f58cee9de2894323ba92dc8d04fc67f0/Media%20ae97551b9771479483c01b8e1dd049ba/Towards%20an%20accurate%20prediction%20of%20the%20thermal%20stab%2046628f9e157c42bbb30f4a0bcc9074d8.md), using the following features
    - amino acid sequences
    - physicochemical features
    - living temp of organism and salt bridges
    - temp dpdt statistical potentials
    - descriptors of protein surface
    - flexibility
    - hydropathy
    - hydrogen bonding
    - packing
    - protein chain lengths ⇒ nope

we noticed a number of problems and issues with ProTherm and therefore
cleaned and pruned the data before developing a novel predictor [(link)](http://structure.bmc.lu.se/VariBench/stability.php).

small impact of individual features, however together they yield rather good performance ⇒ 100 features selected from up to 2000+ features for best performances

Analyses of N-terminal [34] and entire proteomes [35] showed that isoforms often have different cellular stabilities (turnover rates). The **turnover** has strong correlation with **thermal stability.**

- Stability is NOT linked to:
    - to chain length
    - protein sensitivity (pathogenicity)
    - 
- Features:
    - physicochemical
    - structural, and composition features that describe properties of entire proteins
    - protein chain length
    - molecular weight
    - isoelectric point
    - CHNSO (carbon, hydrogen, nitrogen, sulphur, oxygen) counts for elem types & frequencies
    - 6 amino acid group counts and freq
    - count and freq of -/+ charged hydrophilic & hydrophobic residues
    - dipeptide counts
    
    ⇒ step of redundant features removal
    
- Some features computed using:
    - PROFEAT (structural & physicochemical)
    - PROTEIN RECON (protein charge density based electronic porperties)
    - ProtDCal (generate sequence based descriptors)

---

### **Rapid protein stability prediction (**RaSP)
**using deep learning representations (2022)**

- created model used to (quickly) compute all (~9M) stability changes possible for all single amino acid changes in 1382 human proteins
- single amino acid substitutions can have substantial effects on protein
stability depending on:
    - the particular substitution
    - the surrounding atomistic environment
- methods to compute stability changes:
    - Rosetta
    - FoldX
    - methods based on molecular dynamics simulations (Gapsys et al., 2016)

- **Supervised** models: **Problem with systemic biases** from dataset
- **Self-Supervised** models: trained to predict masked amino acid labels ⇒ learned likelihood distribution

**RaSP: 1st step: self supervised model then supervised, with transfer learning**

---

### A Deep-Learning Sequence-Based Method to Predict Protein Stability Changes Upon Genetic Variations

given the symmetry between the two molecular systems M and W, for the reverse variation XM → XW the corresponding change in Gibbs free energy has the opposite sign:
$∆∆GMW = −∆∆GMW$

pre-trained ACDC-NN-Seq using the predictions of another method, DDGun3D

∆∆G variants reported by some of the most widely used datasets extracted from Protherm [25] database, already cleaned for redundancies and inaccuracies that are known to affect this database, which are: S2648 [26 ] and Varibench [27

---

### ThermoMutDB: a thermodynamic database for missense mutations ()

![ThermoMutDB_explained.pdf.png](Resources%20summary%208e1ddd141d1d433a9f8786d5b0eb8f55/ThermoMutDB_explained.pdf.png)

---

### FireProtDB:

![https://loschmidt.chemi.muni.cz/fireprotdb/images/workflow.png](https://loschmidt.chemi.muni.cz/fireprotdb/images/workflow.png)

---

### AI challenges for predicting the impact of mutations on protein stability (2021)

![List of AI-based DDG predictors studied.png](Resources%20summary%208e1ddd141d1d433a9f8786d5b0eb8f55/List_of_AI-based_DDG_predictors_studied.png)

- found that accuracy of the **predictors has stagnated at around 1 kcal/mol** (RMSE)
- methods use information abt prot seq, structure and evolution
- *links to excellent reviews and comparative tests*
- features used:
    - almost all based on the 3D protein structure
    - relative solvent accessible surface area (RSA)
    - changes:
        - in folding free energy (∆∆W)
        - in volume of mutated residue (∆Vol)
        - in residue hydrophobicity (∆Hyd)
    - evolutionnary infos:
        - multiple seq alignments of the query protein
        - substitution matrices (eg. BLOSUM62)
- good results w/ very simple model consisting of a linear combination of 3 features:
    - **RSA of mutated residues**
    - **∆Vol**
    - **∆Hyd**
- FoldX & Rosetta: do a bit worse than AI based methods, could be due to detailed atomic representation which makes them sensitive to resolution defects
- ⇒ accuracy **limit of 1 kcal/mol could be due to**:
    - low number of mutations in the training set
    - fundamental reasons in biology
    - uncontrolled biases in predictors
    - approximations on which predictors are based (wild type structure not mutant one etc.)
    - entropy contributions to the folding free energy overlooked
    - intrinsic errors on experimental ∆∆G values (measures not with same T & pH for ex.)
- boost of prediction accuracy by selecting features that have not yet been combined
- antisymmetry property should be satisfied, see Ssym data set, deviation from antisymmetry δ = ∆∆G(A →B) + ∆∆G(B → A) is an important measure for the evaluation of the lack of bias.
- *talk about unbiased dataset created by Cladararu et al.* Systematic investigation of the data set dependency of protein stability predictors
- *New data have recently been collected:*
    - R. Nikam, A. Kulandaisamy, K. Harini, D. Sharma, M. M. Gromiha, ProThermDB: thermodynamic database for proteins and mutants revisited after 15 years, Nucleic Acids Research 49 (D1) (2021) D420–D424
    - ThermoMutDB
    - J. Stourac, J. Dubrava, M. Musil, J. Horackova, J. Damborsky, S. Mazurenko, D. Bednar, FireProtDB: database of manually curated protein stability data, Nucleic acids research 49 (D1) (2021) D319–D324
- not a lot of data for deep learning, even though pretraining can help prevent overfitting to some extent
- could add Metagenomic sequence (not doable here IMO)

---

### Towards Compilation of Balanced Protein Stability Datasets: Flattening the ∆∆G Curve through Systematic Under-sampling

- created a curated PON-tstab dataset via
    1. clustering mutations based on biochemical and/or structural features 
    2. selecting 3 mutations from every 2kcal/mol of each cluster
    3. Finally systematic under sampling
- result: datasets with a better ∆∆G distributions
- Under-sampling code is available on GitHub ([https://github.com/narodkebabci/gRoR](https://github.com/narodkebabci/gRoR))
- datasets:
    - have a lot of neutral mutations (-0.5 < ∆∆G < 0.5) + more destabilizing mutations (⇒ bias toward destabilization)
    - have a lot more alanine replacement mutations ⇒ more bias
    - Evidently, not only the asymmetry (skewness) but also peaked-ness (kurtosis) of the ∆∆G distributions needs specific attention to compile ideal datasets.

---

### Khan academy: biomolecules ⇒ protein stability

[https://www.khanacademy.org/test-prep/mcat/biomolecules](https://www.khanacademy.org/test-prep/mcat/biomolecules)

![Screenshot from 2022-10-11 17-40-18.png](Resources%20summary%208e1ddd141d1d433a9f8786d5b0eb8f55/Screenshot_from_2022-10-11_17-40-18.png)