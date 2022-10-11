# Main summary

### Features

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
- symmetry $∆∆GMW = −∆∆GMW$

**Step of redundant features removal needs to be done !**

- Some features computed using:
    - PROFEAT (structural & physicochemical)
    - PROTEIN RECON (protein charge density based electronic porperties)
    - ProtDCal (generate sequence based descriptors)

---

### Already used Datasets

- **ThermoMutDB:**
    - includes ProTherm cleaned and pruned [(link)](http://structure.bmc.lu.se/VariBench/stability.php)
        - (S2648, Q3421  and Varibench)
- **FireProtDB**
- Ssym
- PON-stab
- pre-trained ACDC-NN-Seq using the predictions of another method, DDGun3D

**About training/test datasets:**

- [check composition of test and train datasets]

---

### ML implementations

limit of 1 kcal/mol RMSE ⇒ might be hard to go lower (RMSE stagnated in the last 15 years), but **remember the goal is to classify the different mutation between each other, more than computing the correct tm**