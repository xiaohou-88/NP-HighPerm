# NP-HighPerm

**NP-HighPerm** is an assay-aware multimodal dual-task framework for predicting the membrane permeability of nonpeptidic macrocycles (NPMs). The model integrates molecular fingerprints, SMILES-derived sequence representations, molecular graph features, and assay-context information to support both continuous permeability regression and binary permeability classification.

> **Note**  
> This repository is a research release for the NP-HighPerm project. It provides curated data splits, feature engineering scripts, training/evaluation scripts, evaluation metrics, selected model-related files, and experimental results.

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
│   └── best_result/
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
├── test.py
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/NP-HighPerm.git
cd NP-HighPerm
```

### 2. Create a Python environment

```bash
conda create -n np-highperm python=3.10 -y
conda activate np-highperm
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If RDKit cannot be installed through pip, install it using conda:

```bash
conda install -c conda-forge rdkit
```

---

## Dataset

The curated five-fold data splits are stored in:

```text
all_data_split/folds/
```

Each fold contains one training file and one held-out test file:

```text
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
```

The preprocessing procedure includes:

1. Retaining PAMPA and Caco-2 permeability measurements.
2. Removing invalid or implausible permeability values.
3. Applying endpoint-priority deduplication.
4. Representing permeability values as `logPapp`.
5. Clipping permeability values to the range `[-8, -4]`.
6. Assigning binary labels using the threshold `logPapp = -6`.
7. Generating stratified five-fold cross-validation splits.

---

## Model

NP-HighPerm uses three complementary molecular representations.

### 1. Fingerprint branch

The fingerprint branch encodes molecular substructure information using Morgan fingerprints and MACCS keys.

### 2. Sequence branch

The sequence branch encodes SMILES strings using a Transformer-based sequence encoder.

### 3. Graph branch

The graph branch constructs molecular graphs and encodes atom-level connectivity using graph neural network layers.

### 4. Assay-aware fusion

The assay type, such as PAMPA or Caco-2, is used as contextual information to dynamically adjust the contribution of each molecular modality.

### 5. Dual-task prediction

The final prediction head jointly performs:

- regression of continuous membrane permeability values;
- binary classification of permeable and non-permeable compounds.

---

## Usage

### Training

Run the main training script:

```bash
python main.py
```

The script uses predefined five-fold data splits stored in:

```text
all_data_split/folds/
```

### Testing

A testing script is provided as:

```text
test.py
```

Before running the test script, please check the paths of the test CSV file and model checkpoint in `test.py`.

Example command:

```bash
python test.py
```

If needed, modify the following paths inside `test.py`:

```python
TEST_CSV = "all_data_split/folds/fold_1_test.csv"
MODEL_WEIGHTS = "new_result/best_result/fold_1/best_model.pth"
```

The testing script loads the test dataset, initializes the model, loads the saved model weights, and reports the evaluation metrics.

---

## Results

The main experimental results are stored in:

```text
new_result/best_result/
```

The average five-fold results are provided in:

```text
new_result/best_result/average_results.csv
```

Fold-specific results are stored in:

```text
new_result/best_result/fold_1/
new_result/best_result/fold_2/
new_result/best_result/fold_3/
new_result/best_result/fold_4/
new_result/best_result/fold_5/
```

Each fold directory may contain:

```text
detailed_predictions.csv
fold_*_test_results.csv
test_results.csv
train_results.csv
best_model.pth
```

---

## Evaluation Metrics

Regression performance is evaluated using:

- Mean squared error (MSE)
- Root mean squared error (RMSE)
- Coefficient of determination (R²)
- Concordance index (CI)

Classification performance is evaluated using:

- Accuracy
- F1 score
- Area under the receiver operating characteristic curve (AUROC/AUC)
- Area under the precision--recall curve (AUPRC/AP)

---

## Representative Performance

The representative five-fold cross-validation performance of NP-HighPerm is:

| Metric | Performance |
|---|---:|
| MSE | 0.2387 ± 0.0163 |
| RMSE | 0.4884 ± 0.0167 |
| R² | 0.6623 ± 0.0239 |
| CI | 0.8200 ± 0.0067 |
| Accuracy | 0.8498 ± 0.0121 |
| F1 | 0.8547 ± 0.0118 |
| AUROC | 0.9196 ± 0.0046 |

---

## Public Release Scope

This repository currently includes:

- Curated five-fold data splits
- Feature engineering scripts
- Training and testing scripts
- Metric calculation utilities
- Fold-wise prediction results
- Average performance results
- Selected model-related files and checkpoints

Some internal implementation details or selected training weights may be updated in later versions.

---

## Reproducibility Notes

To reproduce or inspect the reported results:

1. Prepare the Python environment.
2. Install dependencies using `requirements.txt`.
3. Confirm that the five-fold split files are available in `all_data_split/folds/`.
4. Run the main training script:

```bash
python main.py
```

5. Run the test script after checking the test data and model checkpoint paths:

```bash
python test.py
```

6. Check result files in:

```text
new_result/best_result/
```

---

## Citation

If you use this repository or the NP-HighPerm results in your research, please cite:

```bibtex
@article{hou2026nphighperm,
  title={NP-HighPerm: An Assay-Aware Multimodal Dual-Task Framework for Nonpeptidic Macrocycle Membrane Permeability Prediction},
  author={},
  journal={Journal Name},
  year={2026},
  note={Manuscript in preparation}
}
```

---

## License

This repository is released for academic and research use. Please refer to the `LICENSE` file for detailed terms.

---

