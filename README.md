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

Dataset

The curated data splits used in this study are stored in:

all_data_split/folds/

Each fold contains one training file and one held-out test file:

fold_1_train.csv
fold_1_test.csv
fold_2_train.csv
fold_2_test.csv
fold_3_train.csv
fold_3_test.csv
fold_4_train.csv
fold_4_test.csv
fold_5_train.csv
fold_5_test.csv

The preprocessing procedure includes:

Retaining PAMPA and Caco-2 permeability measurements.
Removing invalid or implausible permeability values.
Applying endpoint-priority deduplication.
Representing permeability values as logPapp.
Clipping permeability values to the range [-8, -4].
Assigning binary labels using the threshold logPapp = -6.
Generating stratified five-fold cross-validation splits.
Model

NP-HighPerm uses three complementary molecular representations:

1. Fingerprint branch

The fingerprint branch encodes molecular substructure information using Morgan fingerprints and MACCS keys.

2. Sequence branch

The sequence branch encodes SMILES strings using a Transformer-based sequence encoder.

3. Graph branch

The graph branch constructs molecular graphs and encodes atom-level connectivity using graph neural network layers.

4. Assay-aware fusion

The assay type, such as PAMPA or Caco-2, is used as contextual information to dynamically adjust the contribution of each molecular modality.

5. Dual-task prediction

The final prediction head jointly performs:

regression of continuous membrane permeability values;
binary classification of permeable and non-permeable compounds.
Installation

Create a Python environment:

conda create -n np-highperm python=3.10 -y
conda activate np-highperm

Install the main dependencies:

pip install numpy pandas scipy scikit-learn matplotlib seaborn tqdm torch umap-learn

RDKit is recommended for molecular feature generation. If RDKit cannot be installed through pip, install it using conda:

conda install -c conda-forge rdkit
Usage
Run training

The main training entry is:

python main.py

Depending on the local configuration of the scripts, the training process uses the predefined five-fold data splits in:

all_data_split/folds/
Feature engineering

Molecular feature extraction is implemented in:

feature_engineering.py
Model definition

The model architecture is implemented in:

model.py
Training utilities

Training-related functions are implemented in:

train.py
Evaluation metrics

Regression and classification metrics are implemented in:

metrics.py
Results

The main experimental results are stored in:

new_result/01weight_best_result/

The average five-fold results are provided in:

new_result/01weight_best_result/average_results.csv

Fold-specific results are stored in:

new_result/01weight_best_result/fold_1/
new_result/01weight_best_result/fold_2/
new_result/01weight_best_result/fold_3/
new_result/01weight_best_result/fold_4/
new_result/01weight_best_result/fold_5/

Each fold directory may contain:

detailed_predictions.csv
fold_*_test_results.csv
test_results.csv
train_results.csv
Evaluation Metrics

Regression performance is evaluated using:

Mean squared error (MSE)
Root mean squared error (RMSE)
Coefficient of determination (R²)
Concordance index (CI)

Classification performance is evaluated using:

Accuracy
F1 score
Area under the receiver operating characteristic curve (AUROC/AUC)
Area under the precision--recall curve (AUPRC/AP)
Representative Performance

The representative five-fold cross-validation performance of NP-HighPerm is:

Metric	Performance
MSE	0.2387 ± 0.0163
RMSE	0.4884 ± 0.0167
R²	0.6623 ± 0.0239
CI	0.8200 ± 0.0067
Accuracy	0.8498 ± 0.0121
F1	0.8547 ± 0.0118
AUROC	0.9196 ± 0.0046
Public Release Scope

This repository currently includes:

Curated five-fold data splits
Feature engineering scripts
Training and evaluation scripts
Metric calculation utilities
Fold-wise prediction results
Average performance results
Partially released model-related files

Some internal model details or selected training weights may not be fully included in the current public version. Additional materials may be released after manuscript publication or upon reasonable request.

Reproducibility Notes

To reproduce the reported results:

Prepare the Python environment.
Confirm that the five-fold split files are available in all_data_split/folds/.
Run the main script:
python main.py
Check output files in:
new_result/01weight_best_result/

Because this repository may be a partial research release, full training from scratch may require access to the complete internal model implementation and configuration files.

Citation

If you use this repository or the NP-HighPerm results in your research, please cite:

