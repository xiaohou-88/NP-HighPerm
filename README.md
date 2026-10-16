# NP-HighPerm

**NP-HighPerm: An Assay-Aware Multimodal Dual-Task Framework for Nonpeptidic Macrocycle Membrane Permeability Prediction**

NP-HighPerm is a deep learning framework for predicting the membrane permeability of nonpeptidic macrocycles (NPMCs). It combines complementary molecular representations with experimental assay information and jointly performs permeability regression and classification.

## Overview

Nonpeptidic macrocycles occupy a diverse chemical space beyond the conventional rule of five. Predicting their membrane permeability is challenging because experimental measurements come from different assay systems, including PAMPA and Caco-2, and permeability prediction may be required for both quantitative estimation and threshold-based screening.

NP-HighPerm includes:

- **Multimodal molecular representation:** Morgan and MACCS fingerprints, a Transformer-based SMILES encoder, and a graph convolutional network (GCN).
- **Condition-Aware Fusion (CAF):** assay identity is used to generate weights for combining the three molecular representations.
- **Dual-task learning:** continuous permeability regression and auxiliary high-/low-permeability classification are optimized jointly.

## Repository Structure

```text
NP-HighPerm/
├── all_data_split/
│   └── folds/              # Five-fold train/test data
├── split/                  # Dataset splitting utilities
├── visualization/          # Visualization scripts
├── new_result/             # Saved experimental outputs
├── feature_engineering.py
├── main.py
├── metrics.py
├── model.py
├── train.py
├── test.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

Clone the repository and create a Python environment:

```bash
git clone https://github.com/xiaohou-88/NP-HighPerm.git
cd NP-HighPerm
conda create -n np-highperm python=3.10 -y
conda activate np-highperm
pip install -r requirements.txt
```

## Dataset

The study uses data from the **Non-Peptidic Macrocycle Membrane Permeability Database (NPMMPD)**. The curated dataset contains **4,148 NPMCs**, including **3,611 PAMPA** and **537 Caco-2** records.

The continuous prediction target is logarithmic permeability, denoted by `y_reg`. Binary labels are defined using the threshold `y_reg = -6`:

- **High permeability (1):** `y_reg >= -6`
- **Low permeability (0):** `y_reg < -6`

The predefined five-fold train/test files are provided in `all_data_split/folds/`.

## Model Architecture

### Molecular representations

1. **Fingerprint encoder:** combines Morgan fingerprints and MACCS keys through a multilayer perceptron.
2. **SMILES encoder:** uses a Transformer to learn sequence-based molecular representations.
3. **Graph encoder:** uses a GCN to extract features from molecular graphs.

### Condition-Aware Fusion

The CAF module uses an embedding of the assay type (PAMPA or Caco-2) to generate modality weights and combine the three molecular representations.

### Dual-task prediction

The model predicts continuous permeability values and permeability classes. Its joint training objective combines mean squared error (MSE) for regression with binary cross-entropy with logits for classification:

```text
Loss = MSE + 0.1 × BCEWithLogits
```

## Training and Evaluation

NP-HighPerm was evaluated using five-fold cross-validation with validation-based checkpoint selection. For each fold:

1. One fold is held out for testing.
2. A validation subset is selected from 10% of the remaining training data.
3. The model is trained for up to 100 epochs using Adam, with a learning rate of 0.001 and batch size of 128.
4. Early stopping uses a patience of 20 epochs, and the checkpoint with the lowest validation MSE is selected.
5. The selected checkpoint is evaluated on the held-out test fold.

Results are summarized as mean ± standard deviation across the five folds.

### Run training

Before running, set `folds_dir` in `main.py` to the location of the supplied fold CSV files (`all_data_split/folds/`). Then execute:

```bash
python main.py
```

For evaluation using a saved checkpoint, configure the relevant data and checkpoint paths in `test.py`, then run:

```bash
python test.py
```

## Reported Results

The following results are reported in the NP-HighPerm manuscript for five-fold evaluation with validation-based checkpoint selection.

| Metric     |       Mean ± SD |
| ---------- | --------------: |
| MSE ↓      | 0.2459 ± 0.0084 |
| RMSE ↓     | 0.4958 ± 0.0085 |
| R² ↑       | 0.6522 ± 0.0150 |
| CI ↑       | 0.8113 ± 0.0045 |
| Accuracy ↑ | 0.8382 ± 0.0141 |
| F1 ↑       | 0.8420 ± 0.0131 |
| AUROC ↑    | 0.9165 ± 0.0051 |

The manuscript also includes ablation studies, Murcko scaffold-based evaluation, physicochemical and assay-defined subgroup analyses, and decision-boundary analysis.

## Citation

If you use NP-HighPerm in your research, please cite the associated manuscript:

*NP-HighPerm: An Assay-Aware Multimodal Dual-Task Framework for Nonpeptidic Macrocycle Membrane Permeability Prediction.*

Publication details will be added when available.

## License

This repository is distributed under the [MIT License](LICENSE).
