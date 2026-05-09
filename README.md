# NP-HighPerm

**NP-HighPerm** is an assay-aware multimodal dual-task framework for predicting the membrane permeability of nonpeptidic macrocycles (NPMs). The model integrates molecular fingerprints, SMILES-derived sequence representations, molecular graph features, and assay-context information to support both continuous permeability regression and binary permeability classification.

> **Note**  
> This repository is a research release for the NP-HighPerm project. It provides the curated data splits, feature engineering scripts, training/evaluation scripts, evaluation metrics, and experimental results. Some model implementation details and selected training weights may be partially released depending on the project stage.

---

## Overview

Nonpeptidic macrocycles occupy the beyond-rule-of-five chemical space and are promising scaffolds for difficult-to-drug intracellular targets. However, their membrane permeability is difficult to model because experimental measurements are often collected from heterogeneous assays and continuous permeability labels are unevenly distributed.

NP-HighPerm addresses these challenges through:

- **Multimodal molecular representation**
  - Morgan and MACCS fingerprints
  - SMILES sequence encoding
  - Molecular graph encoding

- **Assay-aware fusion**
  - PAMPA and Caco-2 assay information is used as contextual input to recalibrate modality contributions.

- **Dual-task learning**
  - Continuous permeability regression
  - Binary permeability classification using a practical permeability threshold

- **Robust evaluation**
  - Five-fold cross-validation
  - Fold-wise prediction results
  - ROC and PR analysis
  - Subgroup robustness analysis

---

## Repository Structure

```text
NP-HighPerm/
├── all_data_split/
│   └── folds/
│       ├── fold_1_train.csv
│       ├── fold_1_test.csv
│       ├── fold_2_train.csv
│       ├── fold_2_test.csv
│       ├── fold_3_train.csv
│       ├── fold_3_test.csv
│       ├── fold_4_train.csv
│       ├── fold_4_test.csv
│       ├── fold_5_train.csv
│       └── fold_5_test.csv
├── new_result/
│   └── 01weight_best_result/
│       ├── average_results.csv
│       ├── fold_1/
│       ├── fold_2/
│       ├── fold_3/
│       ├── fold_4/
│       └── fold_5/
├── feature_engineering.py
├── main.py
├── metrics.py
├── model.py
├── train.py
├── .gitignore
└── LICENSE
